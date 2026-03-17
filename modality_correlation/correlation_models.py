import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, T, D]
        length = x.size(1)
        x = x + self.pe[:, :length, :]
        return self.dropout(x)


class CorrelationModel(nn.Module):
    def __init__(self,
                 text_in_dim=300,
                 audio_in_dim=74,
                 vision_in_dim=35,
                 d_model=128,
                 num_layers=3,
                 num_heads=4,
                 dim_feedforward=256,
                 dropout=0.1,
                 out_dim=64):
        super(CorrelationModel, self).__init__()

        self.text_fc = nn.Linear(text_in_dim, d_model)
        self.audio_fc = nn.Linear(audio_in_dim, d_model)
        self.vision_fc = nn.Linear(vision_in_dim, d_model)

        self.pos_encoder = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.text_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.audio_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.vision_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.text_out_fc = nn.Linear(d_model, out_dim)
        self.audio_out_fc = nn.Linear(d_model, out_dim)
        self.vision_out_fc = nn.Linear(d_model, out_dim)

    # 在 CorrelationModel 类内部添加这个辅助函数
    def _masked_mean_pool(self, seq, mask):
        if mask is not None:
            valid = (~mask).float().unsqueeze(-1)  # [B, T, 1]
            num = valid.sum(dim=1).clamp(min=1.0)
            return (seq * valid).sum(dim=1) / num  # 输出 [B, D]
        return seq.mean(dim=1)
    
    def forward(
        self,
        text, audio, vision,
        text_pad_mask=None, audio_pad_mask=None, vision_pad_mask=None
    ):
        """
        text/audio/vision: [B, T, Din]
        *_pad_mask: [B, T]  (bool) True 表示 padding 位置，需要 mask 掉
        """

        t_emb = self.text_fc(text)
        a_emb = self.audio_fc(audio)
        v_emb = self.vision_fc(vision)

        # 注意：pos encoding 会把 padding 位置也变成非0，所以必须依赖 pad_mask
        t_emb = self.pos_encoder(t_emb)
        a_emb = self.pos_encoder(a_emb)
        v_emb = self.pos_encoder(v_emb)

        # 修复1：传 src_key_padding_mask
        t_enc = self.text_encoder(t_emb, src_key_padding_mask=text_pad_mask)
        a_enc = self.audio_encoder(a_emb, src_key_padding_mask=audio_pad_mask)
        v_enc = self.vision_encoder(v_emb, src_key_padding_mask=vision_pad_mask)

        _out = self.text_out_fc(t_enc) # [B, T, D]
        a_out = self.audio_out_fc(a_enc)
        v_out = self.vision_out_fc(v_enc)

        # 额外保险：把 padding 位置输出强制置 0
        if text_pad_mask is not None:
            t_out = t_out.masked_fill(text_pad_mask.unsqueeze(-1), 0.0)
        if audio_pad_mask is not None:
            a_out = a_out.masked_fill(audio_pad_mask.unsqueeze(-1), 0.0)
        if vision_pad_mask is not None:
            v_out = v_out.masked_fill(vision_pad_mask.unsqueeze(-1), 0.0)

        # 【消融实验核心修改】：在这里直接将特征压缩为样本级全局特征 [B, D]
        t_global = self._masked_mean_pool(t_out, text_pad_mask)
        a_global = self._masked_mean_pool(a_out, audio_pad_mask)
        v_global = self._masked_mean_pool(v_out, vision_pad_mask)

        return t_global, a_global, v_global
