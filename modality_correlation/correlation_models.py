import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create a position encoding matrix of size [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)  # Use sin for even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # Use cos for odd dimensions
        pe = pe.unsqueeze(0)  # Add batch dimension [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [B, T, D]
        Add positional encoding to x
        """
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
        """
        Parameter description:
        text_in_dim: The dimension of text input features (default 300)
        audio_in_dim: The dimension of audio input features (default 74)
        vision_in_dim: The dimension of vision input features (default 35)
        d_model: Transformer hidden layer dimension
        num_layers: Number of Transformer layers
        num_heads: Number of heads in MultiheadAttention
        dim_feedforward: The dimension of the intermediate layer in the FFN
        dropout: The dropout rate
        out_dim: The dimension of the mapped shared space
        """
        super(CorrelationModel, self).__init__()
        
        # Map to a unified d_model dimension
        self.text_fc = nn.Linear(text_in_dim, d_model)
        self.audio_fc = nn.Linear(audio_in_dim, d_model)
        self.vision_fc = nn.Linear(vision_in_dim, d_model)

        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Define the TransformerEncoderLayer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, 
                                                   nhead=num_heads, 
                                                   dim_feedforward=dim_feedforward, 
                                                   dropout=dropout,
                                                   batch_first=True)
        # Create the TransformerEncoder
        self.text_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.audio_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.vision_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Finally, map to the shared space dimension out_dim
        self.text_out_fc = nn.Linear(d_model, out_dim)
        self.audio_out_fc = nn.Linear(d_model, out_dim)
        self.vision_out_fc = nn.Linear(d_model, out_dim)

    def forward(self, text, audio, vision):
        """
        text:   [B, T_l, text_in_dim]
        audio:  [B, T_a, audio_in_dim]
        vision: [B, T_v, vision_in_dim]
        return:
            t_out: [B, T_l, out_dim]
            a_out: [B, T_a, out_dim]
            v_out: [B, T_v, out_dim]
        """
        # 1) padding mask：True=valid
        text_valid = (text.abs().sum(dim=-1) > 0)    # [B, T_l]
        audio_valid = (audio.abs().sum(dim=-1) > 0)  # [B, T_a]
        vision_valid = (vision.abs().sum(dim=-1) > 0)# [B, T_v]

        # Transformer 的 src_key_padding_mask：True=PAD，需要取反
        text_pad = ~text_valid
        audio_pad = ~audio_valid
        vision_pad = ~vision_valid

        # Linear mapping to d_model dimension
        t_emb = self.text_fc(text)    # [B, T_l, d_model]
        a_emb = self.audio_fc(audio)  # [B, T_a, d_model]
        v_emb = self.vision_fc(vision)# [B, T_v, d_model]

        t_emb = self.pos_encoder(t_emb)
        a_emb = self.pos_encoder(a_emb)
        v_emb = self.pos_encoder(v_emb)

        # 3) encoder（关键：传 mask）
        t_encoded = self.text_encoder(t_emb, src_key_padding_mask=text_pad)
        a_encoded = self.audio_encoder(a_emb, src_key_padding_mask=audio_pad)
        v_encoded = self.vision_encoder(v_emb, src_key_padding_mask=vision_pad)

        # 4) 输出层
        t_out = self.out_fc(t_encoded)
        a_out = self.out_fc(a_encoded)
        v_out = self.out_fc(v_encoded)

        # 5) （强烈建议）把 PAD 位置清零，避免后续 loss/bmm 被污染
        t_out = t_out * text_valid.unsqueeze(-1).type_as(t_out)
        a_out = a_out * audio_valid.unsqueeze(-1).type_as(a_out)
        v_out = v_out * vision_valid.unsqueeze(-1).type_as(v_out)

        return t_out, a_out, v_out
