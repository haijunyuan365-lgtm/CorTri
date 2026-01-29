import torch
from torch import nn
import torch.nn.functional as F
import math
import os
from modules.transformer import TransformerEncoder
from modality_correlation.correlation_models import CorrelationModel
from modules.position_embedding import SinusoidalPositionalEmbedding
# ==============================================================================
# Part 1: 支持 Mask 的 TriSAT 核心组件
# ==============================================================================

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

    def forward(self, query, key, value, key_padding_mask=None, value_padding_mask=None, correlation_bias=None, lambda_param=1.0):
        tgt_len, bsz, embed_dim = query.size()
        src_len_k = key.size(0)
        src_len_v = value.size(0)

        # Projections
        q = F.linear(query, self.in_proj_weight[:embed_dim], self.in_proj_bias[:embed_dim] if self.in_proj_bias is not None else None)
        k = F.linear(key, self.in_proj_weight[embed_dim:2*embed_dim], self.in_proj_bias[embed_dim:2*embed_dim] if self.in_proj_bias is not None else None)
        v = F.linear(value, self.in_proj_weight[2*embed_dim:], self.in_proj_bias[2*embed_dim:] if self.in_proj_bias is not None else None)
        
        q = q * self.scaling

        # [T, B, D] -> [B, Head, T, D] -> [B*Head, T, D]
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(src_len_k, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(src_len_v, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        # 注意：这里不需要再对 k, v 做输入级 masked_fill(0.0) 了，
        # 因为我们会在 attn_weights 层面做更精细的 mask。
        # 当然，为了计算稳定性，保留也无妨，但不起决定性作用。

        # 1. 原始注意力分数 [B*H, T_q, T_k, T_v]
        attn_weights = torch.einsum('iat,ibt,ict->iabc', q, k, v) 

        # 2. 注入物理相关性偏置
        # 注意：Bias 可能在 Padding 处有垃圾值，所以必须在加完 Bias 后再 Mask！
        if correlation_bias is not None:
            bias_expanded = correlation_bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1, 1)
            bias_expanded = bias_expanded.view(bsz * self.num_heads, tgt_len, src_len_k, src_len_v)
            attn_weights = attn_weights + lambda_param * bias_expanded

        # =================================================================
        # [关键修复] 归约前的双重 Masking
        # =================================================================
        
        # 准备 Mask 矩阵 [B*H, T_q, T_k, T_v]
        # 我们主要关心 Key (dim=-2) 的 Mask，因为我们要在这个维度归约
        mask_k_expanded = None
        if key_padding_mask is not None:
            # key_mask: [B, T_k] -> [B*H, 1, T_k, 1] -> expand
            mask_k_expanded = key_padding_mask.view(bsz, 1, src_len_k, 1).repeat(1, self.num_heads, 1, 1).view(bsz * self.num_heads, 1, src_len_k, 1)
            mask_k_expanded = mask_k_expanded.expand(-1, tgt_len, -1, src_len_v)

        # --- 分支 A: 计算 Mean (Sum) ---
        # 对于 Sum，Padding 必须是 0.0
        if mask_k_expanded is not None:
            attn_weights_for_sum = attn_weights.masked_fill(~mask_k_expanded, 0.0)
        else:
            attn_weights_for_sum = attn_weights

        sum_score = torch.sum(attn_weights_for_sum, dim=-2, keepdim=True) # [B*H, T_q, 1, T_v]
        
        # 计算 Scale (分母)
        if key_padding_mask is not None:
            valid_lens_k = key_padding_mask.sum(dim=1).float()
            valid_lens_k = valid_lens_k.masked_fill(valid_lens_k == 0, 1.0)
            scale = valid_lens_k.view(bsz, 1, 1, 1).repeat(1, self.num_heads, tgt_len, src_len_v).view(bsz * self.num_heads, tgt_len, 1, src_len_v)
            avg_score = sum_score / scale
        else:
            avg_score = torch.mean(attn_weights, dim=-2, keepdim=True)

        # --- 分支 B: 计算 Max ---
        # 对于 Max，Padding 必须是 -inf (避免 0.0 大于负数 logits)
        if mask_k_expanded is not None:
            attn_weights_for_max = attn_weights.masked_fill(~mask_k_expanded, -1.0e9)
        else:
            attn_weights_for_max = attn_weights
            
        max_score = torch.max(attn_weights_for_max, dim=-2, keepdim=True)[0]
        
        # 融合
        fused_weights = avg_score + max_score
        fused_weights = fused_weights.squeeze(-2) # [B*Heads, T_q, T_v]

        # =================================================================
        # 输出级 Masking (Value Mask)
        # =================================================================
        if value_padding_mask is not None:
            # mask: [B, T_v] -> [B*H, T_q, T_v]
            mask_v = value_padding_mask.unsqueeze(1).unsqueeze(2)
            mask_v = mask_v.repeat(1, self.num_heads, tgt_len, 1)
            mask_v = mask_v.view(bsz * self.num_heads, tgt_len, src_len_v)
            
            fused_weights = fused_weights.masked_fill(~mask_v, -1e9)

        fused_weights = F.softmax(fused_weights.float(), dim=-1).type_as(fused_weights)
        fused_weights = F.dropout(fused_weights, p=self.attn_dropout, training=self.training)

        # 4. 加权求和 (乘 V)
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
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, x_k, x_v, key_padding_mask=None, value_padding_mask=None, correlation_bias=None, lambda_param=1.0):
        residual = x
        x = self.norm1(x)
        
        x2 = self.self_attn(query=x, key=x_k, value=x_v, 
                            key_padding_mask=key_padding_mask,
                            value_padding_mask=value_padding_mask,
                            correlation_bias=correlation_bias, 
                            lambda_param=lambda_param)
        x = residual + self.dropout1(x2)

        residual = x
        x = self.norm2(x)
        x = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = residual + self.dropout2(x)
        return x

dataset_specific_configs = {
    "mosei_senti": {
        "text_in_dim": 300,
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
        self.embed_positions = SinusoidalPositionalEmbedding(self.d_model)
        
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
        
        self.lambda_param = nn.Parameter(torch.tensor(1.0))

        combined_dim = 2 * self.d_model
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, self.output_dim)
    
    def forward(self, x_l, x_a, x_v):
        # =================================================================
        # [生成 Mask]
        # x_a: [B, T_a, D] -> sum(dim=2) != 0 -> [B, T_a]
        # =================================================================
        mask_l = (x_l.abs().sum(dim=2) > 0)   # [B, T_l]
        mask_a = (x_a.abs().sum(dim=2) > 0)   # [B, T_a]
        mask_v = (x_v.abs().sum(dim=2) > 0)   # [B, T_v]

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
            
            m_l = mask_l.float()
            m_a = mask_a.float()
            m_v = mask_v.float()
            # [B, T_l, T_a]
            C_TA = C_TA * (m_l.unsqueeze(2) * m_a.unsqueeze(1))
            # [B, T_l, T_v]
            C_TV = C_TV * (m_l.unsqueeze(2) * m_v.unsqueeze(1))
            # [B, T_a, T_v]
            C_AV = C_AV * (m_a.unsqueeze(2) * m_v.unsqueeze(1))
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

        # ======================================================
        # [新增] 应用位置编码 (CorMulT 的样式)
        # =====================================================
        if self.embed_positions is not None:
            # proj_l 是 [T, B, D]
            # transpose(0, 1) 变成 [B, T, D]
            # [:, :, 0] 变成 [B, T]，作为 generating positions 的依据
            proj_l = proj_l + self.embed_positions(proj_l.transpose(0, 1)[:, :, 0]).transpose(0, 1)
            proj_a = proj_a + self.embed_positions(proj_a.transpose(0, 1)[:, :, 0]).transpose(0, 1)
            proj_v = proj_v + self.embed_positions(proj_v.transpose(0, 1)[:, :, 0]).transpose(0, 1)
        # ======================================================
        
        # Stream 1: Key=Audio, Value=Video
        # 传入 key_mask=Audio, value_mask=Video
        h_s1 = proj_l
        for layer in self.trisat_stream1:
            h_s1 = layer(h_s1, proj_a, proj_v, 
                         key_padding_mask=mask_a, 
                         value_padding_mask=mask_v,
                         correlation_bias=C_cube_stream1, 
                         lambda_param=self.lambda_param)

        # Stream 2: Key=Video, Value=Audio
        # 传入 key_mask=Video, value_mask=Audio
        h_s2 = proj_l
        for layer in self.trisat_stream2:
            h_s2 = layer(h_s2, proj_v, proj_a, 
                         key_padding_mask=mask_v, 
                         value_padding_mask=mask_a,
                         correlation_bias=C_cube_stream2, 
                         lambda_param=self.lambda_param)

        hs1_pool = masked_mean_pool(h_s1, mask_l)
        hs2_pool = masked_mean_pool(h_s2, mask_l)
        last_hs = torch.cat([hs1_pool, hs2_pool], dim=1)   # [B, 2D]
        
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        
        output = self.out_layer(last_hs_proj)
        
        return output, last_hs
def masked_mean_pool(seq_TBD, mask_BT):
    # seq_TBD: [T, B, D] -> [B, T, D]
    seq = seq_TBD.transpose(0, 1)
    mask = mask_BT.unsqueeze(-1).type_as(seq)   # [B, T, 1]
    denom = mask.sum(dim=1).clamp_min(1.0)      # [B, 1]
    return (seq * mask).sum(dim=1) / denom      # [B, D]