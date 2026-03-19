import torch
from torch import nn
import torch.nn.functional as F
import os
from modality_correlation.correlation_models import CorrelationModel
from modules.position_embedding import SinusoidalPositionalEmbedding

class TrimodalMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_dropout=0., bias=True,
                 use_experiment_d=True, dbg_print=True, dbg_max_batches=5,
                 normalize_kv=True, temperature=5.0, logit_clamp=20.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.use_experiment_d = use_experiment_d   # 保留字段，但 bias 注入改为 idea 的 pre-add
        self.dbg_print = dbg_print
        self.dbg_max_batches = dbg_max_batches
        self._dbg_cnt = 0
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5
        self.normalize_kv = normalize_kv
        self.temperature = float(temperature) if temperature is not None else None
        self.logit_clamp = float(logit_clamp) if logit_clamp is not None else None
        
        # 【修改2：Softmax pooling 的可学习温度系数】
        self.pool_tau = nn.Parameter(torch.tensor(1.0))
        
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
        tgt_len, bsz, embed_dim = query.size()
        src_len_k = key.size(0)
        src_len_v = value.size(0)

        q = F.linear(query, self.in_proj_weight[:embed_dim],
                     self.in_proj_bias[:embed_dim] if self.in_proj_bias is not None else None)
        k = F.linear(key, self.in_proj_weight[embed_dim:2*embed_dim],
                     self.in_proj_bias[embed_dim:2*embed_dim] if self.in_proj_bias is not None else None)
        v = F.linear(value, self.in_proj_weight[2*embed_dim:],
                     self.in_proj_bias[2*embed_dim:] if self.in_proj_bias is not None else None)

        q = q * self.scaling
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)  # [B*H, Tq, d]
        k = k.contiguous().view(src_len_k, bsz * self.num_heads, self.head_dim).transpose(0, 1) # [B*H, Tk, d]
        v = v.contiguous().view(src_len_v, bsz * self.num_heads, self.head_dim).transpose(0, 1) # [B*H, Tv, d]

        if self.normalize_kv:
            k = F.normalize(k, p=2, dim=-1, eps=1e-8)
            v = F.normalize(v, p=2, dim=-1, eps=1e-8)

        # S_sem : [B*H, Tq, Tk, Tv]
        attn_weights = torch.einsum('iat,ibt,ict->iabc', q, k, v)

        # key mask expanded to [B*H, Tq, Tk, Tv] for masking during reduction over Tk
        mask_k_expanded = None
        if key_padding_mask is not None:
            mask_k_expanded = (
                key_padding_mask.view(bsz, 1, src_len_k, 1)
                .repeat(1, self.num_heads, 1, 1)
                .view(bsz * self.num_heads, 1, src_len_k, 1)
                .expand(-1, tgt_len, -1, src_len_v)
            )

        # =========================
        # (IDEA) bias pre-add: S_final = S_sem + lambda * C_cube
        # =========================
        if correlation_bias is not None:
            # correlation_bias: [B, Tq, Tk, Tv] -> expand heads -> [B*H, Tq, Tk, Tv]
            bias_expanded = correlation_bias.unsqueeze(1).repeat(1, self.num_heads, 1, 1, 1)
            bias_expanded = bias_expanded.view(bsz * self.num_heads, tgt_len, src_len_k, src_len_v)

            attn_weights = attn_weights + lambda_param * bias_expanded

            if self.dbg_print and self._dbg_cnt < self.dbg_max_batches:
                with torch.no_grad():
                    bw = (lambda_param * bias_expanded)
                    print("[DBG-IDEA] attn_weights(mean/std/maxabs)=",
                          attn_weights.mean().item(), attn_weights.std().item(), attn_weights.abs().max().item())
                    print("[DBG-IDEA] lambda*bias(mean/std/maxabs)=",
                          bw.mean().item(), bw.std().item(), bw.abs().max().item())
                    if torch.is_tensor(lambda_param):
                        print("[DBG-IDEA] lambda =", float(lambda_param.detach().item()))
                self._dbg_cnt += 1

        # =========================
        # 【修改2：Softmax-weighted pooling over K (dim=-2) -> weights over V】
        # 替代原始易受 Outlier 干扰的 avg+max 操作
        # =========================
        if mask_k_expanded is not None:
            attn_for_pool = attn_weights.masked_fill(~mask_k_expanded, -1e9)
            attn_valid = attn_weights.masked_fill(~mask_k_expanded, 0.0) # 保证 pad 处不会贡献 sum
        else:
            attn_for_pool = attn_weights
            attn_valid = attn_weights

        # w = softmax(C / \tau)
        tau = torch.clamp(self.pool_tau, min=0.01) # 保证温度系数不为0
        pool_weights = F.softmax(attn_for_pool / tau, dim=-2)
        
        # C_pooled = \sum w * C
        fused_weights = (pool_weights * attn_valid).sum(dim=-2) # 降维至 [B*H, Tq, Tv]

        # temperature + clamp
        if (self.temperature is not None) and (self.temperature != 1.0):
            fused_weights = fused_weights / self.temperature
        if self.logit_clamp is not None:
            fused_weights = fused_weights.clamp(-self.logit_clamp, self.logit_clamp)

        # mask V positions
        if value_padding_mask is not None:
            mask_v = value_padding_mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,Tv]
            mask_v = mask_v.repeat(1, self.num_heads, tgt_len, 1).view(bsz * self.num_heads, tgt_len, src_len_v)
            fused_weights = fused_weights.masked_fill(~mask_v, -1e9)

        fused_weights = F.softmax(fused_weights.float(), dim=-1).type_as(v)
        fused_weights = F.dropout(fused_weights, p=self.attn_dropout, training=self.training)

        # output on V (unaligned-friendly)
        attn = torch.bmm(fused_weights, v)  # [B*H, Tq, d]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
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
    seq = seq_TBD.transpose(0, 1)
    mask = mask_BT.unsqueeze(-1).type_as(seq)
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

        self.use_correlation = getattr(hyp_params, 'use_correlation', False)

        self.attn_dropout = hyp_params.attn_dropout
        self.embed_dropout = hyp_params.embed_dropout
        self.out_dropout = hyp_params.out_dropout

        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_model, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_model, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_model, kernel_size=1, padding=0, bias=False)

        self.missing_a = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.missing_v = nn.Parameter(torch.zeros(1, 1, self.d_model))
        nn.init.normal_(self.missing_a, mean=0.0, std=0.02)
        nn.init.normal_(self.missing_v, mean=0.0, std=0.02)

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
                print("WARNING: No pretrained path found. Using initialized weights.")

            for param in self.corr_model.parameters():
                param.requires_grad = False
            self.corr_model.eval()
            print("Correlation model loaded and frozen.")

            # 【修改1：方案B】为每个模态添加轻量级的 Projection Adapter
            # 初始化时让最后一层权重和偏置为0，保证初始的输出完全等同于原本的 frozen 输出 (F_T_adp = F_T_pp + 0)
            dim_out = corr_config['out_dim']
            self.adapter_t = nn.Sequential(
                nn.Linear(dim_out, dim_out),
                nn.ReLU(),
                nn.Linear(dim_out, dim_out)
            )
            self.adapter_a = nn.Sequential(
                nn.Linear(dim_out, dim_out),
                nn.ReLU(),
                nn.Linear(dim_out, dim_out)
            )
            self.adapter_v = nn.Sequential(
                nn.Linear(dim_out, dim_out),
                nn.ReLU(),
                nn.Linear(dim_out, dim_out)
            )
            for m in [self.adapter_t, self.adapter_a, self.adapter_v]:
                nn.init.zeros_(m[-1].weight)
                nn.init.zeros_(m[-1].bias)

        else:
            self.corr_model = None
            print("Running in STANDARD mode (No correlation model loaded).")

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

        self.w_tv = nn.Parameter(torch.tensor(0.33))
        self.w_ta = nn.Parameter(torch.tensor(0.33))
        self.w_va = nn.Parameter(torch.tensor(0.33))
        self.w_av = nn.Parameter(torch.tensor(0.33))
        
        # 【修改3：乘积项权重】
        self.w_inter_s1 = nn.Parameter(torch.tensor(0.1))
        self.w_inter_s2 = nn.Parameter(torch.tensor(0.1))
        
        self.lambda_param = nn.Parameter(torch.tensor(1.0))

        combined_dim = 2 * self.d_model
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, self.output_dim)

    def _add_positional(self, proj_TBD, mask_BT):
        if self.embed_positions is None:
            return proj_TBD
        T, B, D = proj_TBD.size()
        device = proj_TBD.device
        positions = torch.arange(1, T + 1, device=device).unsqueeze(0).expand(B, T).long()
        if mask_BT is not None:
            positions = positions * mask_BT.long()
        pos_emb = self.embed_positions(positions)
        return proj_TBD + pos_emb.transpose(0, 1)

    def _inject_missing_token_TBD(self, proj_TBD, mask_BT, missing_token_T1BD):
        T, B, D = proj_TBD.size()
        device = proj_TBD.device
        token = missing_token_T1BD.expand(1, B, D).to(device)
        proj_TBD_p1 = torch.cat([proj_TBD, token], dim=0)
        if mask_BT is None:
            mask_BT_p1 = torch.ones(B, T + 1, device=device, dtype=torch.bool)
            injected = torch.zeros(B, device=device, dtype=torch.bool)
            return proj_TBD_p1, mask_BT_p1, injected
        valid_counts = mask_BT.sum(dim=1)
        empty = (valid_counts == 0)
        mask_BT_p1 = torch.cat([mask_BT, torch.zeros(B, 1, device=device, dtype=torch.bool)], dim=1)
        if empty.any():
            mask_BT_p1[empty, -1] = True
        return proj_TBD_p1, mask_BT_p1, empty

    def _expand_cube_for_missing(self, cube, add_to_k=False, add_to_v=False):
        if cube is None:
            return None
        B, Tq, Tk, Tv = cube.shape
        device = cube.device
        dtype = cube.dtype
        out = cube
        if add_to_k:
            pad_k = torch.zeros(B, Tq, 1, Tv, device=device, dtype=dtype)
            out = torch.cat([out, pad_k], dim=2)
            Tk = Tk + 1
        if add_to_v:
            pad_v = torch.zeros(B, Tq, Tk, 1, device=device, dtype=dtype)
            out = torch.cat([out, pad_v], dim=3)
        return out

    def forward(self, x_l, x_a, x_v):
        mask_l = (x_l.abs().sum(dim=2) > 0)
        mask_a = (x_a.abs().sum(dim=2) > 0)
        mask_v = (x_v.abs().sum(dim=2) > 0)

        C_cube_stream1 = None
        C_cube_stream2 = None

        if self.use_correlation:
            use_bias = True

            # 【修改3】加入 inter 项联合 softmax
            w_s1 = torch.softmax(torch.stack([self.w_tv, self.w_ta, self.w_va, self.w_inter_s1]), dim=0)
            w_s2 = torch.softmax(torch.stack([self.w_tv, self.w_ta, self.w_av, self.w_inter_s2]), dim=0)
            lam = torch.sigmoid(self.lambda_param)

            with torch.no_grad():
                self.corr_model.eval()
                # corr_model 的 TransformerEncoder 依赖 pad_mask，否则 padding token 会污染有效 token
                text_pad = ~mask_l
                audio_pad = ~mask_a
                vision_pad = ~mask_v
                F_T_pp, F_A_pp, F_V_pp = self.corr_model(
                    x_l, x_a, x_v,
                    text_pad_mask=text_pad,
                    audio_pad_mask=audio_pad,
                    vision_pad_mask=vision_pad,
                )

            # 【修改1：方案B】让特征经过可学习的残差 Adapter
            F_T_adp = F_T_pp + self.adapter_t(F_T_pp)
            F_A_adp = F_A_pp + self.adapter_a(F_A_pp)
            F_V_adp = F_V_pp + self.adapter_v(F_V_pp)

            F_T_norm = F.normalize(F_T_adp, p=2, dim=-1)
            F_A_norm = F.normalize(F_A_adp, p=2, dim=-1)
            F_V_norm = F.normalize(F_V_adp, p=2, dim=-1)

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

            R_TV_1 = C_TV.unsqueeze(2)  # [B, Tq, 1, Tv]
            R_TA_1 = C_TA.unsqueeze(3)  # [B, Tq, Tk, 1]
            R_AV_1 = C_AV.unsqueeze(1)  # [B, 1, Tk, Tv]
            
            # 【修改3】多模态乘积交互项 delta * (C_TV * C_TA)
            inter_1 = R_TV_1 * R_TA_1
            C_cube_stream1 = w_s1[0] * R_TV_1 + w_s1[1] * R_TA_1 + w_s1[2] * R_AV_1 + w_s1[3] * inter_1

            R_TV_2 = C_TV.unsqueeze(3)
            R_TA_2 = C_TA.unsqueeze(2)
            R_VA_2 = C_VA.unsqueeze(1)
            
            inter_2 = R_TV_2 * R_TA_2
            C_cube_stream2 = w_s2[0] * R_TV_2 + w_s2[1] * R_TA_2 + w_s2[2] * R_VA_2 + w_s2[3] * inter_2
        else:
            use_bias = False
            C_cube_stream1 = None
            C_cube_stream2 = None
            lam = 0.0

        x_l_p = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_a_p = x_a.transpose(1, 2)
        x_v_p = x_v.transpose(1, 2)

        proj_l = self.proj_l(x_l_p).permute(2, 0, 1)
        proj_a = self.proj_a(x_a_p).permute(2, 0, 1)
        proj_v = self.proj_v(x_v_p).permute(2, 0, 1)

        proj_l = self._add_positional(proj_l, mask_l)
        proj_a = self._add_positional(proj_a, mask_a)
        proj_v = self._add_positional(proj_v, mask_v)

        proj_a, mask_a_p1, empty_a = self._inject_missing_token_TBD(proj_a, mask_a, self.missing_a)
        proj_v, mask_v_p1, empty_v = self._inject_missing_token_TBD(proj_v, mask_v, self.missing_v)

        mask_a = mask_a_p1
        mask_v = mask_v_p1

        if use_bias:
            C_cube_stream1 = self._expand_cube_for_missing(C_cube_stream1, add_to_k=True, add_to_v=True)
            C_cube_stream2 = self._expand_cube_for_missing(C_cube_stream2, add_to_k=True, add_to_v=True)

        h_s1 = proj_l
        for layer in self.trisat_stream1:
            h_s1 = layer(
                h_s1, proj_a, proj_v,
                key_padding_mask=mask_a,
                value_padding_mask=mask_v,
                correlation_bias=C_cube_stream1,
                lambda_param=lam
            )

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
        return output, last_hs