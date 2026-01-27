# 文件: src/models.py
import torch
from torch import nn
import torch.nn.functional as F
import math
import os
from modules.transformer import TransformerEncoder
from modality_correlation.correlation_models import CorrelationModel

class TrimodalMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_dropout=0., bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = nn.Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.Tensor(3 * embed_dim)) if bias else None
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, correlation_bias=None, lambda_param=1.0):
        tgt_len, bsz, embed_dim = query.size()
        src_len_k = key.size(0)
        src_len_v = value.size(0)

        q = F.linear(query, self.in_proj_weight[:embed_dim], self.in_proj_bias[:embed_dim] if self.in_proj_bias is not None else None)
        k = F.linear(key, self.in_proj_weight[embed_dim:2*embed_dim], self.in_proj_bias[embed_dim:2*embed_dim] if self.in_proj_bias is not None else None)
        v = F.linear(value, self.in_proj_weight[2*embed_dim:], self.in_proj_bias[2*embed_dim:] if self.in_proj_bias is not None else None)
        
        q = q * self.scaling

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(src_len_k, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(src_len_v, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        # S_ijk = sum(Q_in * K_jn * V_kn)
        attn_weights = torch.einsum('iat,ibt,ict->iabc', q, k, v) 

        # 注入物理相关性偏置
        if correlation_bias is not None:
            bias_expanded = correlation_bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1, 1)
            # 确保 bias 形状匹配 [B*H, T_q, T_k, T_v]
            bias_expanded = bias_expanded.view(bsz * self.num_heads, tgt_len, src_len_k, src_len_v)
            attn_weights = attn_weights + lambda_param * bias_expanded

        # TriSAT Fusion: Mean + Max
        avg_score = torch.mean(attn_weights, dim=-1, keepdim=True)
        max_score = torch.max(attn_weights, dim=-1, keepdim=True)[0]
        fused_weights = avg_score + max_score
        fused_weights = fused_weights.squeeze(-1) # [B*Heads, T_q, T_k]

        fused_weights = F.softmax(fused_weights.float(), dim=-1).type_as(fused_weights)
        fused_weights = F.dropout(fused_weights, p=self.attn_dropout, training=self.training)

        # =================================================================
        # [关键修改] 改回标准的 Attention 逻辑：乘 v
        # =================================================================
        # fused_weights: [B*H, T_q(50), T_k(400)]
        # v:             [B*H, T_v(400), D]
        # Result:        [B*H, T_q(50), D]  <-- 维度完美匹配！
        attn = torch.bmm(fused_weights, v)
        
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        return attn

class TriSATEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_dropout=0.1, dropout=0.1):
        super().__init__()
        self.self_attn = TrimodalMultiheadAttention(embed_dim, num_heads, attn_dropout=attn_dropout)
        
        self.linear1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(4 * embed_dim, embed_dim)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        # 兼容性处理：如果只传入一个 tensor，也要能处理
        # TriSAT 中 x_k, x_v 来源不同，所以 norm 应该分开定义，或者共享参数取决于设计
        # 这里假设输入已经 Norm 过，或者由 Layer 内部 Norm
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, x_k, x_v, correlation_bias=None, lambda_param=1.0):
        residual = x
        x = self.norm1(x)
        # 注意: x_k, x_v 通常是其他模态，是否Norm取决于架构，TriSAT源码里似乎是在输入前Norm的
        # 为了保险，我们在这里对 Key/Value 也做个 Norm (如果维度相同)
        # 但考虑到 x_k/x_v 可能已经经过了 layer norm，这里简单起见只 norm x (Query)
        
        x2 = self.self_attn(query=x, key=x_k, value=x_v, correlation_bias=correlation_bias, lambda_param=lambda_param)
        x = residual + self.dropout1(x2)

        residual = x
        x = self.norm2(x)
        x = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = residual + self.dropout2(x)
        return x

dataset_specific_configs = {
    "mosei_senti": {
        "text_in_dim": 768,
        "audio_in_dim": 74,
        "vision_in_dim": 35,
        "d_model": 128,
        "num_layers": 3,
        "num_heads": 4,
        "dim_feedforward": 256,
        "dropout": 0.1,
        "out_dim": 64,
    },
    "ch_sims": {
        "text_in_dim": 768,
        "audio_in_dim": 25,
        "vision_in_dim": 177,
        "d_model": 128,
        "num_layers": 3,
        "num_heads": 4,
        "dim_feedforward": 256,
        "dropout": 0.1,
        "out_dim": 64,
    }
}

class MULTModel(nn.Module):
    def __init__(self, hyp_params):
        super(MULTModel, self).__init__()
        self.orig_d_l, self.orig_d_a, self.orig_d_v = hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v
        
        self.d_l = self.d_a = self.d_v = self.d_model = 30 
        
        self.num_heads = hyp_params.num_heads
        self.layers = hyp_params.layers
        self.output_dim = hyp_params.output_dim
        self.use_correlation = hyp_params.use_correlation
        self.attn_dropout = hyp_params.attn_dropout
        self.embed_dropout = hyp_params.embed_dropout
        self.out_dropout = hyp_params.out_dropout

        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_model, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_model, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_model, kernel_size=1, padding=0, bias=False)

        if self.use_correlation:
            corr_config = dataset_specific_configs[hyp_params.dataset].copy()
            corr_config['text_in_dim'] = hyp_params.orig_d_l
            corr_config['audio_in_dim'] = hyp_params.orig_d_a
            corr_config['vision_in_dim'] = hyp_params.orig_d_v
            
            self.corr_model = CorrelationModel(**corr_config)
            
            if hasattr(hyp_params, 'corr_model_path') and os.path.exists(hyp_params.corr_model_path):
                print(f"Loading pretrained correlation model from {hyp_params.corr_model_path}")
                self.corr_model.load_state_dict(torch.load(hyp_params.corr_model_path, map_location='cpu'))
            else:
                print("No pretrained path found. Training from SCRATCH.")
                
        self.trisat_stream1 = nn.ModuleList([
            TriSATEncoderLayer(self.d_model, self.num_heads, self.attn_dropout) 
            for _ in range(self.layers)
        ])
        
        self.trisat_stream2 = nn.ModuleList([
            TriSATEncoderLayer(self.d_model, self.num_heads, self.attn_dropout)
            for _ in range(self.layers)
        ])

        self.w_tv = nn.Parameter(torch.tensor(0.33))
        self.w_ta = nn.Parameter(torch.tensor(0.33))
        self.w_va = nn.Parameter(torch.tensor(0.33))
        self.w_av = nn.Parameter(torch.tensor(0.33))
        
        # =================================================================
        # [关键修改] 初始 lambda 改为 1.0
        # =================================================================
        # 既然在 Stage 2 我们冻结了 Correlation Model，说明我们信任它。
        # 让模型一上来就完全接收 Bias，能加速 Unaligned 数据的对齐学习。
        self.lambda_param = nn.Parameter(torch.tensor(1.0))

        combined_dim = 2 * self.d_model
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, self.output_dim)

    def forward(self, x_l, x_a, x_v):
        B = x_l.size(0)
        C_cube_stream1 = None 
        C_cube_stream2 = None
        
        if self.use_correlation:
            with torch.no_grad():
                self.corr_model.eval()
                F_T_pp, F_A_pp, F_V_pp = self.corr_model(x_l, x_a, x_v)

            F_T_norm = F.normalize(F_T_pp, p=2, dim=-1)
            F_A_norm = F.normalize(F_A_pp, p=2, dim=-1)
            F_V_norm = F.normalize(F_V_pp, p=2, dim=-1)

            C_TA = torch.bmm(F_T_norm, F_A_norm.transpose(1, 2))
            C_TV = torch.bmm(F_T_norm, F_V_norm.transpose(1, 2))
            C_AV = torch.bmm(F_A_norm, F_V_norm.transpose(1, 2)) 
            C_VA = C_AV.transpose(1, 2)

            R_TV_1 = C_TV.unsqueeze(2) 
            R_TA_1 = C_TA.unsqueeze(3)
            R_AV_1 = C_AV.unsqueeze(1)
            C_cube_stream1 = self.w_tv * R_TV_1 + self.w_ta * R_TA_1 + self.w_va * R_AV_1

            R_TV_2 = C_TV.unsqueeze(3)
            R_TA_2 = C_TA.unsqueeze(2)
            R_VA_2 = C_VA.unsqueeze(1)
            C_cube_stream2 = self.w_tv * R_TV_2 + self.w_ta * R_TA_2 + self.w_av * R_VA_2

        x_l_p = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_a_p = x_a.transpose(1, 2)
        x_v_p = x_v.transpose(1, 2)

        proj_l = self.proj_l(x_l_p).permute(2, 0, 1)
        proj_a = self.proj_a(x_a_p).permute(2, 0, 1)
        proj_v = self.proj_v(x_v_p).permute(2, 0, 1)

        h_s1 = proj_l
        for layer in self.trisat_stream1:
            h_s1 = layer(h_s1, proj_a, proj_v, 
                         correlation_bias=C_cube_stream1, 
                         lambda_param=self.lambda_param)

        h_s2 = proj_l
        for layer in self.trisat_stream2:
            h_s2 = layer(h_s2, proj_v, proj_a, 
                         correlation_bias=C_cube_stream2, 
                         lambda_param=self.lambda_param)

        last_hs = torch.cat([h_s1[-1], h_s2[-1]], dim=1)
        
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        
        output = self.out_layer(last_hs_proj)
        
        return output, last_hs