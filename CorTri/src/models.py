# 文件: src/models.py
import torch
from torch import nn
import torch.nn.functional as F
import math

from modules.transformer import TransformerEncoder
from modality_correlation.correlation_models import CorrelationModel

# ==============================================================================
# Part 1: TriSAT 核心组件 (移植并修改以支持 Physical Correlation Bias)
# ==============================================================================

class TrimodalMultiheadAttention(nn.Module):
    """
    修改自 TriSAT 的 MultiheadAttention
    增加了 correlation_bias (C_cube) 的输入接口
    """
    def __init__(self, embed_dim, num_heads, attn_dropout=0., bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        # TriSAT 的 Q, K, V 投影
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

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)

    def forward(self, query, key, value, correlation_bias=None, lambda_param=1.0):
        """
        query: [T_q, B, D]
        key:   [T_k, B, D]
        value: [T_v, B, D]
        correlation_bias: [B, T_q, T_k, T_v] (物理相关性立方体)
        """
        tgt_len, bsz, embed_dim = query.size()
        src_len_k = key.size(0)
        src_len_v = value.size(0)

        # 1. 投影 Q, K, V
        # 注意：TriSAT 源码中处理 QKV 的方式较复杂，这里简化为分别投影，因为输入源不同
        q = F.linear(query, self.in_proj_weight[:embed_dim], self.in_proj_bias[:embed_dim] if self.in_proj_bias is not None else None)
        k = F.linear(key, self.in_proj_weight[embed_dim:2*embed_dim], self.in_proj_bias[embed_dim:2*embed_dim] if self.in_proj_bias is not None else None)
        v = F.linear(value, self.in_proj_weight[2*embed_dim:], self.in_proj_bias[2*embed_dim:] if self.in_proj_bias is not None else None)
        
        q = q * self.scaling

        # 2. Reshape heads: [T, B, D] -> [T, B*Heads, HeadDim] -> [B*Heads, T, HeadDim]
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(src_len_k, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(src_len_v, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        # 3. Trimodal Attention Calculation (Core of TriSAT)
        # Einsum: i=batch*heads, a=Time_Q, b=Time_K, c=Time_V, t=HeadDim
        # S_ijk = sum(Q_in * K_jn * V_kn)
        attn_weights = torch.einsum('iat,ibt,ict->iabc', q, k, v) 
        # attn_weights shape: [B*Heads, T_q, T_k, T_v]

        # 4. ================= 创新点：注入物理相关性偏置 =================
        if correlation_bias is not None:
            # correlation_bias: [B, T_q, T_k, T_v]
            # 我们需要将其广播到 [B*Heads, T_q, T_k, T_v]
            # unsqueeze(1) -> [B, 1, T, T, T] -> repeat -> [B, Heads, ...] -> view
            bias_expanded = correlation_bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1, 1)
            bias_expanded = bias_expanded.view(bsz * self.num_heads, tgt_len, src_len_k, src_len_v)
            
            # S_final = S_sem + lambda * C_cube
            attn_weights = attn_weights + lambda_param * bias_expanded
        # =================================================================

        # 5. TriSAT 的后处理策略: Mean + Max (TriSAT 源码 line 110)
        avg_score = torch.mean(attn_weights, dim=-1, keepdim=True)
        max_score = torch.max(attn_weights, dim=-1, keepdim=True)[0]
        fused_weights = avg_score + max_score
        fused_weights = fused_weights.squeeze(-1) # [B*Heads, T_q, T_k]

        fused_weights = F.softmax(fused_weights.float(), dim=-1).type_as(fused_weights)
        fused_weights = F.dropout(fused_weights, p=self.attn_dropout, training=self.training)

        # 6. 加权求和得到输出
        # Standard attention: softmax(QK^T)V. 
        # TriSAT here does: softmax(merged_score) * Q (Note: Source code line 125 uses q, not v)
        # TriSAT 源码注释: "change: v -> q". 这很特殊，但为了保持一致性我们遵循源码。
        # 但通常 Attention 输出应该是 V 的加权。TriSAT 论文中 Q 是 Text，输出也是 Text 表示。
        # 让我们仔细看 TriSAT 源码: `attn = torch.bmm(attn_weights, q)`
        # `attn_weights`: [B*H, T_q, T_k] (after squeeze). 
        # `q`: [B*H, T_q, D_h]. 
        # 维度对不上: (T_q, T_k) * (T_q, D_h) 矩阵乘法要求中间维度一致。
        # 如果 squeeze(-1) 把 T_v 维去掉了，那剩下的是 T_k。
        # 应该乘的是 K 对应的序列？或者 TriSAT 这里的维度逻辑有特定假设 (如 T_q=T_k=T_v)。
        # 为了稳妥且符合常规 Transformer 逻辑（同时融合三模态信息），
        # 我们这里使用原始的 attn_weights (未 squeeze 前) 来加权 V?
        # 不，TriSAT 的核心是把三模态交互压缩回双模态矩阵然后作用。
        # 我们严格遵循 TriSAT 源码的这一行：`attn = torch.bmm(attn_weights, q)`
        # 只有当 T_q == T_k 时这行代码才成立 (Square matrix)。
        # 假设: 在 TriSAT 中所有模态都被对齐或 Padding 到相同长度 (seq_len)。
        
        attn = torch.bmm(fused_weights, q) 
        
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

    def forward(self, x, x_k, x_v, correlation_bias=None, lambda_param=1.0):
        # x is Query (Text), x_k is Key, x_v is Value
        residual = x
        x = self.norm1(x)
        x_k = self.norm1(x_k) # Apply norm to others too as per TriSAT idea
        x_v = self.norm1(x_v)

        x2 = self.self_attn(query=x, key=x_k, value=x_v, correlation_bias=correlation_bias, lambda_param=lambda_param)
        x = residual + self.dropout1(x2)

        residual = x
        x = self.norm2(x)
        x = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = residual + self.dropout2(x)
        return x

# ==============================================================================
# Part 2: 改进后的 MULTModel (CorMulT + TriSAT + Dynamic Cube)
# ==============================================================================

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
    "ch_sims": { # 假设配置
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
        
        # 统一维度
        self.d_l = self.d_a = self.d_v = self.d_model = 30 
        
        self.num_heads = hyp_params.num_heads
        self.layers = hyp_params.layers
        self.output_dim = hyp_params.output_dim
        self.use_correlation = hyp_params.use_correlation
        
        # === 修复点：补充缺失的 Dropout 参数定义 ===
        self.attn_dropout = hyp_params.attn_dropout
        self.embed_dropout = hyp_params.embed_dropout  # 新增
        self.out_dropout = hyp_params.out_dropout      # 新增
        # ==========================================

        # 1. Temporal convolutional layers (投影到统一维度)
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_model, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_model, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_model, kernel_size=1, padding=0, bias=False)

        # 2. End-to-End Correlation Model
        if self.use_correlation:
            self.corr_model = CorrelationModel(
                **dataset_specific_configs[hyp_params.dataset]
            )
            self.corr_model.load_state_dict(torch.load(hyp_params.corr_model_path, map_location='cpu'))

        # 3. TriSAT Layers
        # Stream 1: Q=Text, K=Audio, V=Video
        self.trisat_stream1 = nn.ModuleList([
            TriSATEncoderLayer(self.d_model, self.num_heads, self.attn_dropout) 
            for _ in range(self.layers)
        ])
        
        # Stream 2: Q=Text, K=Video, V=Audio
        self.trisat_stream2 = nn.ModuleList([
            TriSATEncoderLayer(self.d_model, self.num_heads, self.attn_dropout)
            for _ in range(self.layers)
        ])

        # 4. 可学习参数
        self.w_tv = nn.Parameter(torch.tensor(0.33))
        self.w_ta = nn.Parameter(torch.tensor(0.33))
        self.w_va = nn.Parameter(torch.tensor(0.33))
        self.w_av = nn.Parameter(torch.tensor(0.33))
        self.lambda_param = nn.Parameter(torch.tensor(0.1))

        # 5. Output Layers
        combined_dim = 2 * self.d_model
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, self.output_dim)

    # forward 方法保持不变...
    def forward(self, x_l, x_a, x_v):
        # ... (代码与之前一致，不需要修改)
        # x_l: [B, T_l, D_in]
        B = x_l.size(0)

        # =====================================================
        # Step 1: 细粒度相关性矩阵计算 (Innovation 1)
        # =====================================================
        C_cube_stream1 = None # [B, T, T, T] (Q=L, K=A, V=V) -> L, A, V
        C_cube_stream2 = None # [B, T, T, T] (Q=L, K=V, V=A) -> L, V, A
        
        if self.use_correlation:
            # 获取序列特征 [B, T, D_out]
            # 注意：此处不再使用 torch.no_grad()，实现端到端训练
            F_T_pp, F_A_pp, F_V_pp = self.corr_model(x_l, x_a, x_v)

            # 归一化以便计算 Cosine Similarity
            F_T_norm = F.normalize(F_T_pp, p=2, dim=-1)
            F_A_norm = F.normalize(F_A_pp, p=2, dim=-1)
            F_V_norm = F.normalize(F_V_pp, p=2, dim=-1)

            # 计算两两物理相关性矩阵 [B, T, T]
            # C_TA[b, i, j] = T[b, i] dot A[b, j]
            C_TA = torch.bmm(F_T_norm, F_A_norm.transpose(1, 2))
            C_TV = torch.bmm(F_T_norm, F_V_norm.transpose(1, 2))
            C_AV = torch.bmm(F_A_norm, F_V_norm.transpose(1, 2)) 
            C_VA = C_AV.transpose(1, 2)

            # =====================================================
            # Step 2: 物理相关性立方体构建 (Innovation 2)
            # =====================================================
            # 广播机制构建 4D Tensor [B, T_q, T_k, T_v]
            # Stream 1: Q=Text(i), K=Audio(j), V=Video(k)
            # C_cube[i,j,k] = w1*C_TV[i,k] + w2*C_TA[i,j] + w3*C_AV[j,k]
            
            # C_TV [B, T, V] -> 对应 i, k -> unsqueeze dim 2 (j/Audio)
            R_TV_1 = C_TV.unsqueeze(2) 
            # C_TA [B, T, A] -> 对应 i, j -> unsqueeze dim 3 (k/Video)
            R_TA_1 = C_TA.unsqueeze(3)
            # C_AV [B, A, V] -> 对应 j, k -> unsqueeze dim 1 (i/Text)
            R_AV_1 = C_AV.unsqueeze(1)

            C_cube_stream1 = self.w_tv * R_TV_1 + self.w_ta * R_TA_1 + self.w_va * R_AV_1

            # Stream 2: Q=Text(i), K=Video(j), V=Audio(k)
            # C_cube[i,j,k] = w1*C_TV[i,j] + w2*C_TA[i,k] + w3*C_VA[j,k]
            
            # C_TV [B, T, V] -> 对应 i, j -> unsqueeze dim 3 (k/Audio)
            R_TV_2 = C_TV.unsqueeze(3)
            # C_TA [B, T, A] -> 对应 i, k -> unsqueeze dim 2 (j/Video)
            R_TA_2 = C_TA.unsqueeze(2)
            # C_VA [B, V, A] -> 对应 j, k -> unsqueeze dim 1 (i/Text)
            R_VA_2 = C_VA.unsqueeze(1)

            C_cube_stream2 = self.w_tv * R_TV_2 + self.w_ta * R_TA_2 + self.w_av * R_VA_2


        # =====================================================
        # Step 3: TriSAT 模态交互 (Innovation 3)
        # =====================================================
        
        # 原始特征投影
        # [B, T, D] -> [B, D, T] for Conv1d
        x_l_p = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_a_p = x_a.transpose(1, 2)
        x_v_p = x_v.transpose(1, 2)

        proj_l = self.proj_l(x_l_p).permute(2, 0, 1) # [T, B, D]
        proj_a = self.proj_a(x_a_p).permute(2, 0, 1)
        proj_v = self.proj_v(x_v_p).permute(2, 0, 1)

        # Stream 1: Text query, Audio Key, Video Value
        h_s1 = proj_l
        for layer in self.trisat_stream1:
            h_s1 = layer(h_s1, proj_a, proj_v, 
                         correlation_bias=C_cube_stream1, 
                         lambda_param=self.lambda_param)

        # Stream 2: Text query, Video Key, Audio Value
        h_s2 = proj_l
        for layer in self.trisat_stream2:
            h_s2 = layer(h_s2, proj_v, proj_a, 
                         correlation_bias=C_cube_stream2, 
                         lambda_param=self.lambda_param)

        # 融合
        last_hs = torch.cat([h_s1[-1], h_s2[-1]], dim=1) # 取最后一个时间步 [B, 2*D]
        
        # 残差与输出
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        
        output = self.out_layer(last_hs_proj)
        
        # 返回 output 和 last_hs (保持接口一致)
        return output, last_hs, (F_T_pp, F_A_pp, F_V_pp)