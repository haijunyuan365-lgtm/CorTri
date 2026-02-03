import torch
from torch import nn
import torch.nn.functional as F
import os
from modality_correlation.correlation_models import CorrelationModel
from modules.position_embedding import SinusoidalPositionalEmbedding

# ==============================================================================
# Part 1: 支持 Mask 的 TriSAT 核心组件 (含实验 D：late injection)
# ==============================================================================

class TrimodalMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_dropout=0., bias=True,
                 use_experiment_d=True, dbg_print=True, dbg_max_batches=5,
                 # ===== 改法1新增：防止 logits 饱和 =====
                 normalize_kv=True, temperature=5.0, logit_clamp=20.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.use_experiment_d = use_experiment_d

        self.dbg_print = dbg_print
        self.dbg_max_batches = dbg_max_batches
        self._dbg_cnt = 0

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        # ===== 改法1新增 =====
        self.normalize_kv = normalize_kv
        self.temperature = float(temperature) if temperature is not None else None
        self.logit_clamp = float(logit_clamp) if logit_clamp is not None else None

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

    def forward(self, query, key, value,
                key_padding_mask=None, value_padding_mask=None,
                correlation_bias=None, lambda_param=1.0):
        """
        query: [Tq, B, D]
        key:   [Tk, B, D]
        value: [Tv, B, D]
        key_padding_mask:   [B, Tk]  True=valid
        value_padding_mask: [B, Tv]  True=valid
        correlation_bias:   [B, Tq, Tk, Tv]
        """
        tgt_len, bsz, embed_dim = query.size()
        src_len_k = key.size(0)
        src_len_v = value.size(0)

        # Projections
        q = F.linear(query, self.in_proj_weight[:embed_dim],
                     self.in_proj_bias[:embed_dim] if self.in_proj_bias is not None else None)
        k = F.linear(key, self.in_proj_weight[embed_dim:2*embed_dim],
                     self.in_proj_bias[embed_dim:2*embed_dim] if self.in_proj_bias is not None else None)
        v = F.linear(value, self.in_proj_weight[2*embed_dim:],
                     self.in_proj_bias[2*embed_dim:] if self.in_proj_bias is not None else None)

        q = q * self.scaling

        # [T, B, D] -> [B*H, T, Dh]
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(src_len_k, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(src_len_v, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        # ===== 改法1新增：对 K/V 做归一化，抑制 tri-linear logits 爆炸 =====
        if self.normalize_kv:
            k = F.normalize(k, p=2, dim=-1, eps=1e-8)
            v = F.normalize(v, p=2, dim=-1, eps=1e-8)

                # 1) 4D tri-attn logits: [B*H, Tq, Tk, Tv]
        #    端到端训练更容易在 tri-linear einsum 溢出，因此这里强制用 fp32 计算，再在输出处 cast 回原 dtype。
        orig_dtype = query.dtype
        q = q.float()
        k = k.float()
        v = v.float()

        attn_weights = torch.einsum('iat,ibt,ict->iabc', q, k, v)

        # 2) 构造 key mask expand: [B*H, Tq, Tk, Tv]
        mask_k_expanded = None
        if key_padding_mask is not None:
            mask_k_expanded = (
                key_padding_mask.view(bsz, 1, src_len_k, 1)
                .repeat(1, self.num_heads, 1, 1)
                .view(bsz * self.num_heads, 1, src_len_k, 1)
                .expand(-1, tgt_len, -1, src_len_v)
            )

        # ==============================
        # 原版（early injection）：bias 加在 4D logits 上
        # ==============================
        if (not self.use_experiment_d) and (correlation_bias is not None):
            bias_expanded = correlation_bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1, 1)
            bias_expanded = bias_expanded.view(bsz * self.num_heads, tgt_len, src_len_k, src_len_v)
            attn_weights = attn_weights + lambda_param * bias_expanded

        # 3) 对 attn_weights 做 key 维归约前的双分支 masking
        # --- Sum/Mean 分支：padding=0 ---
        if mask_k_expanded is not None:
            attn_for_sum = attn_weights.masked_fill(~mask_k_expanded, 0.0)
        else:
            attn_for_sum = attn_weights

        sum_score = attn_for_sum.sum(dim=-2, keepdim=True)  # [B*H, Tq, 1, Tv]

        if key_padding_mask is not None:
            valid_lens_k = key_padding_mask.sum(dim=1).float().clamp_min(1.0)  # [B]
            scale = (
                valid_lens_k.view(bsz, 1, 1, 1)
                .repeat(1, self.num_heads, tgt_len, src_len_v)
                .view(bsz * self.num_heads, tgt_len, 1, src_len_v)
            )
            avg_score = sum_score / scale
        else:
            avg_score = attn_weights.mean(dim=-2, keepdim=True)

        # --- Max 分支：padding=finite min（避免 AMP/fp16 下 -1e9 -> -inf 导致 softmax NaN） ---
        neg_large_4d = torch.finfo(attn_weights.dtype).min
        if mask_k_expanded is not None:
            attn_for_max = attn_weights.masked_fill(~mask_k_expanded, neg_large_4d)
        else:
            attn_for_max = attn_weights

        max_score = attn_for_max.max(dim=-2, keepdim=True)[0]  # [B*H, Tq, 1, Tv]

        # 4) 融合得到 fused_weights: [B*H, Tq, Tv]
        fused_weights = (avg_score + max_score).squeeze(-2)

        # ==============================
        # 实验 D（late injection）：bias 先同样 reduce，再加到 fused_weights 上
        # ==============================
        if self.use_experiment_d and (correlation_bias is not None):
            bias_expanded = correlation_bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1, 1)
            bias_expanded = bias_expanded.view(bsz * self.num_heads, tgt_len, src_len_k, src_len_v)

            if mask_k_expanded is not None:
                bias_for_sum = bias_expanded.masked_fill(~mask_k_expanded, 0.0)
                bias_sum = bias_for_sum.sum(dim=-2)  # [B*H, Tq, Tv]

                valid_lens_k = key_padding_mask.sum(dim=1).float().clamp_min(1.0)
                scale2 = (
                    valid_lens_k.view(bsz, 1, 1)
                    .repeat(1, self.num_heads, tgt_len * src_len_v)
                    .view(bsz * self.num_heads, tgt_len, src_len_v)
                )
                bias_avg = bias_sum / scale2

                bias_for_max = bias_expanded.masked_fill(~mask_k_expanded, neg_large_4d)
                bias_max = bias_for_max.max(dim=-2)[0]  # [B*H, Tq, Tv]
            else:
                bias_avg = bias_expanded.mean(dim=-2)
                bias_max = bias_expanded.max(dim=-2)[0]

            bias_fused = bias_avg + bias_max  # [B*H, Tq, Tv]
            fused_weights = fused_weights + lambda_param * bias_fused

            if self.dbg_print and self._dbg_cnt < self.dbg_max_batches:
                print("[DBG-D] fused_weights(mean/std/maxabs)=",
                      fused_weights.mean().item(), fused_weights.std().item(), fused_weights.abs().max().item())
                print("[DBG-D] bias_fused(mean/std/maxabs)=",
                      bias_fused.mean().item(), bias_fused.std().item(), bias_fused.abs().max().item())
                print("[DBG-D] lambda =", float(lambda_param))

                self._dbg_cnt += 1

        # ===== 改法1新增：temperature 缩放 + clamp（在 value mask 之前做，避免把 padding 的极小值 clamp 掉）=====
        if (self.temperature is not None) and (self.temperature != 1.0):
            fused_weights = fused_weights / self.temperature

        if self.logit_clamp is not None:
            fused_weights = fused_weights.clamp(-self.logit_clamp, self.logit_clamp)

        # 5) Value mask（输出级）
        all_invalid = None
        if value_padding_mask is not None:
            mask_v = value_padding_mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,Tv]
            mask_v = mask_v.repeat(1, self.num_heads, tgt_len, 1).view(bsz * self.num_heads, tgt_len, src_len_v)
            neg_large_3d = torch.finfo(fused_weights.dtype).min
            fused_weights = fused_weights.masked_fill(~mask_v, neg_large_3d)
            all_invalid = (~mask_v).all(dim=-1, keepdim=True)  # [B*H, Tq, 1]

        # 6) softmax -> dropout（NaN/Inf safe）
        fw = torch.nan_to_num(fused_weights, nan=0.0,
                              posinf=float(self.logit_clamp) if self.logit_clamp is not None else 20.0,
                              neginf=-(float(self.logit_clamp) if self.logit_clamp is not None else 20.0))
        fw = fw - fw.max(dim=-1, keepdim=True).values
        fused_weights = torch.softmax(fw, dim=-1)

        if all_invalid is not None and all_invalid.any():
            fused_weights = fused_weights.masked_fill(all_invalid, 0.0)

        fused_weights = F.dropout(fused_weights, p=self.attn_dropout, training=self.training)

        # 7) 加权求和： [B*H,Tq,Tv] x [B*H,Tv,Dh] -> [B*H,Tq,Dh]
        attn = torch.bmm(fused_weights, v)

        # reshape back: [Tq, B, D]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn).to(orig_dtype)
        return attn


class TriSATEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_dropout=0.1, dropout=0.1,
                 use_experiment_d=True, dbg_print=True):
        super().__init__()
        self.self_attn = TrimodalMultiheadAttention(
            embed_dim, num_heads, attn_dropout=attn_dropout,
            use_experiment_d=use_experiment_d, dbg_print=dbg_print
        )

        self.linear1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(4 * embed_dim, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, x_k, x_v,
                key_padding_mask=None, value_padding_mask=None,
                correlation_bias=None, lambda_param=1.0):
        residual = x
        x = self.norm1(x)

        x2 = self.self_attn(
            query=x, key=x_k, value=x_v,
            key_padding_mask=key_padding_mask,
            value_padding_mask=value_padding_mask,
            correlation_bias=correlation_bias,
            lambda_param=lambda_param
        )
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


def masked_mean_pool(seq_TBD, mask_BT):
    seq = seq_TBD.transpose(0, 1)  # [B,T,D]
    mask = mask_BT.unsqueeze(-1).type_as(seq)   # [B,T,1]
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (seq * mask).sum(dim=1) / denom


class MULTModel(nn.Module):
    def __init__(self, hyp_params, use_experiment_d=True, dbg_print=True):
        super(MULTModel, self).__init__()
        self.orig_d_l, self.orig_d_a, self.orig_d_v = hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v

        self.d_l = self.d_a = self.d_v = self.d_model = 30
        self.num_heads = hyp_params.num_heads
        self.layers = hyp_params.layers
        self.output_dim = hyp_params.output_dim
        self.use_correlation = hyp_params.use_correlation
        # E2E controls
        self.corr_bias_grad = getattr(hyp_params, 'corr_bias_grad', False)
        self.freeze_corr_model = getattr(hyp_params, 'freeze_corr_model', False)

        self.attn_dropout = hyp_params.attn_dropout
        self.embed_dropout = hyp_params.embed_dropout
        self.out_dropout = hyp_params.out_dropout

        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_model, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_model, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_model, kernel_size=1, padding=0, bias=False)

        # ---- Fix: handle samples with empty (all-zero) modality by injecting a learnable [MISSING] token ----
        # These tokens are appended to A/V sequences when that modality is fully padded for a sample.
        # NOTE: We'll implement injection in forward AFTER proj to match d_model.
        self.missing_a = nn.Parameter(torch.zeros(1, 1, self.d_model))  # [T=1, B=1, D]
        self.missing_v = nn.Parameter(torch.zeros(1, 1, self.d_model))  # [T=1, B=1, D]
        nn.init.normal_(self.missing_a, mean=0.0, std=0.02)
        nn.init.normal_(self.missing_v, mean=0.0, std=0.02)

        # 位置编码
        self.embed_positions = SinusoidalPositionalEmbedding(self.d_model)

        if self.use_correlation:
            corr_config = dataset_specific_configs[hyp_params.dataset].copy()
            corr_config['text_in_dim'] = hyp_params.orig_d_l
            corr_config['audio_in_dim'] = hyp_params.orig_d_a
            corr_config['vision_in_dim'] = hyp_params.orig_d_v

            self.corr_model = CorrelationModel(**corr_config)
            if hasattr(hyp_params, 'corr_model_path') and hyp_params.corr_model_path and os.path.exists(hyp_params.corr_model_path):
                print(f"Loading pretrained correlation model from {hyp_params.corr_model_path}")
                self.corr_model.load_state_dict(torch.load(hyp_params.corr_model_path, map_location='cpu'))
            else:
                print("No pretrained path found. Training from SCRATCH.")

            # Optionally freeze correlation model (no gradient updates)
            if self.freeze_corr_model:
                for p in self.corr_model.parameters():
                    p.requires_grad = False
                print("[Info] corr_model frozen (freeze_corr_model=True)")

        self.trisat_stream1 = nn.ModuleList([
            TriSATEncoderLayer(self.d_model, self.num_heads, self.attn_dropout,
                               use_experiment_d=use_experiment_d, dbg_print=dbg_print)
            for _ in range(self.layers)
        ])
        self.trisat_stream2 = nn.ModuleList([
            TriSATEncoderLayer(self.d_model, self.num_heads, self.attn_dropout,
                               use_experiment_d=use_experiment_d, dbg_print=dbg_print)
            for _ in range(self.layers)
        ])

        # 可学习权重 & λ（注意：softmax/sigmoid 在 forward 里算）
        self.w_tv = nn.Parameter(torch.tensor(0.33))
        self.w_ta = nn.Parameter(torch.tensor(0.33))
        self.w_va = nn.Parameter(torch.tensor(0.33))
        self.w_av = nn.Parameter(torch.tensor(0.33))
        self.lambda_param = nn.Parameter(torch.tensor(2.0))

        combined_dim = 2 * self.d_model
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, self.output_dim)

    def _add_positional(self, proj_TBD, mask_BT):
        """
        proj_TBD: [T,B,D]
        mask_BT:  [B,T] True=valid
        """
        if self.embed_positions is None:
            return proj_TBD
        T, B, D = proj_TBD.size()
        device = proj_TBD.device

        positions = torch.arange(1, T + 1, device=device).unsqueeze(0).expand(B, T).long()
        if mask_BT is not None:
            positions = positions * mask_BT.long()

        pos_emb = self.embed_positions(positions)  # [B,T,D]
        return proj_TBD + pos_emb.transpose(0, 1)

    def _inject_missing_token_TBD(self, proj_TBD, mask_BT, missing_token_T1BD):
        """
        proj_TBD: [T,B,D]
        mask_BT:  [B,T] True=valid
        missing_token_T1BD: [1,1,D] parameter
        Returns:
            proj_Tp1_BD: [T+1,B,D] (append at end)
            mask_BT_p1:  [B,T+1] with last position possibly activated for empty samples
            injected_B:  [B] bool, True if this sample was empty and we activated the missing token
        """
        T, B, D = proj_TBD.size()
        device = proj_TBD.device

        # append token (always)
        token = missing_token_T1BD.expand(1, B, D).to(device)  # [1,B,D]
        proj_TBD_p1 = torch.cat([proj_TBD, token], dim=0)      # [T+1,B,D]

        if mask_BT is None:
            # If no mask provided, assume all valid -> no empty sample
            mask_BT_p1 = torch.ones(B, T + 1, device=device, dtype=torch.bool)
            injected = torch.zeros(B, device=device, dtype=torch.bool)
            return proj_TBD_p1, mask_BT_p1, injected

        # identify empty samples
        valid_counts = mask_BT.sum(dim=1)     # [B]
        empty = (valid_counts == 0)

        # extend mask with a new column for token
        mask_BT_p1 = torch.cat([mask_BT, torch.zeros(B, 1, device=device, dtype=torch.bool)], dim=1)
        # for empty samples, activate the missing token position
        if empty.any():
            mask_BT_p1[empty, -1] = True

        return proj_TBD_p1, mask_BT_p1, empty

    def _expand_cube_for_missing(self, cube, add_to_k=False, add_to_v=False):
        """
        cube: [B, Tq, Tk, Tv] or None
        add_to_k: whether Tk increases by 1
        add_to_v: whether Tv increases by 1
        Return: expanded cube with zeros in the new rows/cols (neutral bias)
        """
        if cube is None:
            return None
        B, Tq, Tk, Tv = cube.shape
        device = cube.device
        dtype = cube.dtype

        out = cube
        if add_to_k:
            pad_k = torch.zeros(B, Tq, 1, Tv, device=device, dtype=dtype)
            out = torch.cat([out, pad_k], dim=2)  # Tk+1
            Tk = Tk + 1
        if add_to_v:
            pad_v = torch.zeros(B, Tq, Tk, 1, device=device, dtype=dtype)
            out = torch.cat([out, pad_v], dim=3)  # Tv+1

        return out

    def forward(self, x_l, x_a, x_v):
        # masks: True=valid
        mask_l = (x_l.abs().sum(dim=2) > 0)
        mask_a = (x_a.abs().sum(dim=2) > 0)
        mask_v = (x_v.abs().sum(dim=2) > 0)

        seq_features = None
        C_cube_stream1 = None
        C_cube_stream2 = None

        # NOTE: keep your existing bias switch
        use_bias = True

        # w softmax；λ sigmoid
        w_s1 = torch.softmax(torch.stack([self.w_tv, self.w_ta, self.w_va]), dim=0)
        w_s2 = torch.softmax(torch.stack([self.w_tv, self.w_ta, self.w_av]), dim=0)
        lam = torch.sigmoid(self.lambda_param) if use_bias else 0.0

        if self.use_correlation and use_bias:
            # corr_model forward
            F_T_pp, F_A_pp, F_V_pp = self.corr_model(x_l, x_a, x_v)
            # For correlation bias, optionally detach to save memory (default: detach)
            if not self.corr_bias_grad:
                F_T_b, F_A_b, F_V_b = F_T_pp.detach(), F_A_pp.detach(), F_V_pp.detach()
            else:
                F_T_b, F_A_b, F_V_b = F_T_pp, F_A_pp, F_V_pp
            seq_features = (F_T_pp, F_A_pp, F_V_pp)

            F_T_norm = F.normalize(F_T_b, p=2, dim=-1)
            F_A_norm = F.normalize(F_A_b, p=2, dim=-1)
            F_V_norm = F.normalize(F_V_b, p=2, dim=-1)

            C_TA = torch.bmm(F_T_norm, F_A_norm.transpose(1, 2))
            C_TV = torch.bmm(F_T_norm, F_V_norm.transpose(1, 2))
            C_AV = torch.bmm(F_A_norm, F_V_norm.transpose(1, 2))
            C_VA = C_AV.transpose(1, 2)

            m_l = mask_l.float()
            m_a = mask_a.float()
            m_v = mask_v.float()

            C_TA = C_TA * (m_l.unsqueeze(2) * m_a.unsqueeze(1))
            C_TV = C_TV * (m_l.unsqueeze(2) * m_v.unsqueeze(1))
            C_AV = C_AV * (m_a.unsqueeze(2) * m_v.unsqueeze(1))

            # cube1: [B, T_l, T_a, T_v]
            R_TV_1 = C_TV.unsqueeze(2)   # [B, T_l, 1, T_v]
            R_TA_1 = C_TA.unsqueeze(3)   # [B, T_l, T_a, 1]
            R_AV_1 = C_AV.unsqueeze(1)   # [B, 1,   T_a, T_v]
            C_cube_stream1 = w_s1[0] * R_TV_1 + w_s1[1] * R_TA_1 + w_s1[2] * R_AV_1

            # cube2: [B, T_l, T_v, T_a] （key=V, value=A）
            R_TV_2 = C_TV.unsqueeze(3)   # [B, T_l, T_v, 1]
            R_TA_2 = C_TA.unsqueeze(2)   # [B, T_l, 1,   T_a]
            R_VA_2 = C_VA.unsqueeze(1)   # [B, 1,   T_v, T_a]
            C_cube_stream2 = w_s2[0] * R_TV_2 + w_s2[1] * R_TA_2 + w_s2[2] * R_VA_2

        # proj
        x_l_p = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_a_p = x_a.transpose(1, 2)
        x_v_p = x_v.transpose(1, 2)

        proj_l = self.proj_l(x_l_p).permute(2, 0, 1)  # [T,B,D]
        proj_a = self.proj_a(x_a_p).permute(2, 0, 1)
        proj_v = self.proj_v(x_v_p).permute(2, 0, 1)

        # 加位置编码（更标准的 positions）
        proj_l = self._add_positional(proj_l, mask_l)
        proj_a = self._add_positional(proj_a, mask_a)
        proj_v = self._add_positional(proj_v, mask_v)

        # ----------------------------------------------------------------------
        # NEW: Inject [MISSING] token for empty A/V samples to avoid "all-masked" -1e9 chain
        # We always append one token; only empty samples activate it in the mask.
        # Also expand correlation cubes accordingly (neutral zeros).
        # ----------------------------------------------------------------------
        # stream1 uses key=A (Tk), value=V (Tv)
        proj_a, mask_a_p1, empty_a = self._inject_missing_token_TBD(proj_a, mask_a, self.missing_a)
        proj_v, mask_v_p1, empty_v = self._inject_missing_token_TBD(proj_v, mask_v, self.missing_v)

        # update masks
        mask_a = mask_a_p1
        mask_v = mask_v_p1

        # expand cubes if they exist and bias is enabled
        if use_bias:
            # C_cube_stream1: [B, T_l, T_a, T_v] -> add both Tk and Tv by 1
            C_cube_stream1 = self._expand_cube_for_missing(C_cube_stream1, add_to_k=True, add_to_v=True)
            # C_cube_stream2: [B, T_l, T_v, T_a] -> key=V, value=A
            C_cube_stream2 = self._expand_cube_for_missing(C_cube_stream2, add_to_k=True, add_to_v=True)

        # stream1: key=A, value=V
        h_s1 = proj_l
        for layer in self.trisat_stream1:
            h_s1 = layer(
                h_s1, proj_a, proj_v,
                key_padding_mask=mask_a,
                value_padding_mask=mask_v,
                correlation_bias=C_cube_stream1,
                lambda_param=lam
            )

        # stream2: key=V, value=A
        h_s2 = proj_l
        for layer in self.trisat_stream2:
            h_s2 = layer(
                h_s2, proj_v, proj_a,
                key_padding_mask=mask_v,
                value_padding_mask=mask_a,
                correlation_bias=C_cube_stream2,
                lambda_param=lam
            )

        hs1_pool = masked_mean_pool(h_s1, mask_l)
        hs2_pool = masked_mean_pool(h_s2, mask_l)
        last_hs = torch.cat([hs1_pool, hs2_pool], dim=1)

        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj = last_hs_proj + last_hs

        output = self.out_layer(last_hs_proj)
        if seq_features is not None:
            return output, last_hs, seq_features
        return output, last_hs
