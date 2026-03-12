import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import argparse
import sys
import os
from tqdm import tqdm

# 添加项目根目录
sys.path.append(os.getcwd())

from modality_correlation.correlation_dataset import UnifiedMultimodalDataset
from modality_correlation.correlation_models import CorrelationModel
from modality_correlation.correlation_loss import TripleLoss

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
    },
    "mosi": {
        "text_in_dim": 768,
        "audio_in_dim": 5,   # 从 check_pkl 中看到的 5
        "vision_in_dim": 20, # 从 check_pkl 中看到的 20
        "d_model": 128,
        "num_layers": 3,
        "num_heads": 4,
        "dim_feedforward": 256,
        "dropout": 0.1,
        "out_dim": 64,
    }
}

def get_collate_fn(max_len=400):
    """
    返回 padding 后的 tensor + 对应 pad_mask（True 表示 padding）
    """
    def _pad_and_mask(seqs):
        # seqs: list of [Ti, D]
        lens = torch.tensor([s.size(0) for s in seqs], dtype=torch.long)  # [B]
        padded = pad_sequence(seqs, batch_first=True)  # [B, Tm, D]
        B, Tm = padded.size(0), padded.size(1)
        ar = torch.arange(Tm).unsqueeze(0).expand(B, Tm)
        pad_mask = ar >= lens.unsqueeze(1)  # True 是 padding
        return padded, pad_mask

    def _safe_trunc(x):
        # x: [T, D]
        if x is None:
            return x
        if x.size(0) == 0:
            # 极端情况：空序列，给 1 个全 0 token，避免 pad_sequence 报错
            return torch.zeros(1, x.size(-1), dtype=x.dtype)
        return x[:max_len]

    def collate_fn(batch):
        # batch item: ((meta, text, audio, vision), (text_neg, audio_neg, vision_neg), label, META)

        texts     = [_safe_trunc(item[0][1]) for item in batch]
        audios    = [_safe_trunc(item[0][2]) for item in batch]
        visions   = [_safe_trunc(item[0][3]) for item in batch]

        texts_neg   = [_safe_trunc(item[1][0]) for item in batch]
        audios_neg  = [_safe_trunc(item[1][1]) for item in batch]
        visions_neg = [_safe_trunc(item[1][2]) for item in batch]

        texts, text_pad = _pad_and_mask(texts)
        audios, audio_pad = _pad_and_mask(audios)
        visions, vision_pad = _pad_and_mask(visions)

        texts_neg, textn_pad = _pad_and_mask(texts_neg)
        audios_neg, audion_pad = _pad_and_mask(audios_neg)
        visions_neg, visionn_pad = _pad_and_mask(visions_neg)

        # 返回：数据 + mask（共 12 个）
        return (texts, audios, visions,
                texts_neg, audios_neg, visions_neg,
                text_pad, audio_pad, vision_pad,
                textn_pad, audion_pad, visionn_pad)

    return collate_fn

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Start Stage 1 Training for {args.dataset_name}...")
    
    # 智能解析你命令行传入的完整 pkl 路径
    if args.data_path.endswith('.pkl'):
        dataset_dir = os.path.dirname(args.data_path)
        pkl_name = os.path.basename(args.data_path)
    else:
        dataset_dir = args.data_path
        pkl_name = "mosi-unaligned_50.pkl"
        
    train_data = UnifiedMultimodalDataset(
        dataset_path=dataset_dir,
        data=args.dataset_name,
        split_type='train',
        if_align=False,
        max_samples=args.max_samples,
        for_correlation=True,               
        perturbation_ratio=args.perturbation_ratio,
        pkl_filename=pkl_name    # <--- 传递解析出的文件名
    )
    valid_data = UnifiedMultimodalDataset(
        dataset_path=dataset_dir,
        data=args.dataset_name,
        split_type='valid',
        if_align=False,
        max_samples=args.max_samples,
        for_correlation=True,
        perturbation_ratio=0.0,  
        pkl_filename=pkl_name    # <--- 传递解析出的文件名
    )

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=get_collate_fn(max_len=args.max_len),
        num_workers=4
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=get_collate_fn(max_len=args.max_len),
        num_workers=4
    )

    config = dataset_specific_configs[args.dataset_name]
    model = CorrelationModel(**config).to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = TripleLoss(margin=args.margin)

    os.makedirs('pre_trained_models', exist_ok=True)
    save_path = f'pre_trained_models/{args.model_save_name}.pt'
    best_val_loss = float('inf')

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs}"):

            (text, audio, vision,
             text_n, audio_n, vision_n,
             text_pad, audio_pad, vision_pad,
             textn_pad, audion_pad, visionn_pad) = batch

            # move to device
            text = text.to(device); audio = audio.to(device); vision = vision.to(device)
            text_n = text_n.to(device); audio_n = audio_n.to(device); vision_n = vision_n.to(device)

            text_pad = text_pad.to(device); audio_pad = audio_pad.to(device); vision_pad = vision_pad.to(device)
            textn_pad = textn_pad.to(device); audion_pad = audion_pad.to(device); visionn_pad = visionn_pad.to(device)

            optimizer.zero_grad()

            # 修复1：forward 传 pad mask
            f_t, f_a, f_v = model(
                text, audio, vision,
                text_pad_mask=text_pad, audio_pad_mask=audio_pad, vision_pad_mask=vision_pad
            )
            f_t_n, f_a_n, f_v_n = model(
                text_n, audio_n, vision_n,
                text_pad_mask=textn_pad, audio_pad_mask=audion_pad, vision_pad_mask=visionn_pad
            )

            # 修复2：loss 也传 mask（anchor/pos/neg 分别对应）
            loss = 0.0
            # Anchor: Audio
            loss += criterion(f_a, f_t, f_t_n, anchor_pad_mask=audio_pad, pos_pad_mask=text_pad,  neg_pad_mask=textn_pad)
            loss += criterion(f_a, f_v, f_v_n, anchor_pad_mask=audio_pad, pos_pad_mask=vision_pad, neg_pad_mask=visionn_pad)
            # Anchor: Text
            loss += criterion(f_t, f_a, f_a_n, anchor_pad_mask=text_pad, pos_pad_mask=audio_pad,  neg_pad_mask=audion_pad)
            loss += criterion(f_t, f_v, f_v_n, anchor_pad_mask=text_pad, pos_pad_mask=vision_pad, neg_pad_mask=visionn_pad)
            # Anchor: Vision
            loss += criterion(f_v, f_a, f_a_n, anchor_pad_mask=vision_pad, pos_pad_mask=audio_pad,  neg_pad_mask=audion_pad)
            loss += criterion(f_v, f_t, f_t_n, anchor_pad_mask=vision_pad, pos_pad_mask=text_pad,  neg_pad_mask=textn_pad)

            loss = loss / 6.0

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / max(1, len(train_loader))

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                (text, audio, vision,
                 text_n, audio_n, vision_n,
                 text_pad, audio_pad, vision_pad,
                 textn_pad, audion_pad, visionn_pad) = batch

                text = text.to(device); audio = audio.to(device); vision = vision.to(device)
                text_n = text_n.to(device); audio_n = audio_n.to(device); vision_n = vision_n.to(device)

                text_pad = text_pad.to(device); audio_pad = audio_pad.to(device); vision_pad = vision_pad.to(device)
                textn_pad = textn_pad.to(device); audion_pad = audion_pad.to(device); visionn_pad = visionn_pad.to(device)

                f_t, f_a, f_v = model(
                    text, audio, vision,
                    text_pad_mask=text_pad, audio_pad_mask=audio_pad, vision_pad_mask=vision_pad
                )
                f_t_n, f_a_n, f_v_n = model(
                    text_n, audio_n, vision_n,
                    text_pad_mask=textn_pad, audio_pad_mask=audion_pad, vision_pad_mask=visionn_pad
                )

                loss = 0.0
                loss += criterion(f_a, f_t, f_t_n, anchor_pad_mask=audio_pad, pos_pad_mask=text_pad,  neg_pad_mask=textn_pad)
                loss += criterion(f_a, f_v, f_v_n, anchor_pad_mask=audio_pad, pos_pad_mask=vision_pad, neg_pad_mask=visionn_pad)

                loss += criterion(f_t, f_a, f_a_n, anchor_pad_mask=text_pad, pos_pad_mask=audio_pad,  neg_pad_mask=audion_pad)
                loss += criterion(f_t, f_v, f_v_n, anchor_pad_mask=text_pad, pos_pad_mask=vision_pad, neg_pad_mask=visionn_pad)

                loss += criterion(f_v, f_a, f_a_n, anchor_pad_mask=vision_pad, pos_pad_mask=audio_pad,  neg_pad_mask=audion_pad)
                loss += criterion(f_v, f_t, f_t_n, anchor_pad_mask=vision_pad, pos_pad_mask=text_pad,  neg_pad_mask=textn_pad)

                val_loss += (loss / 6.0).item()

        avg_val_loss = val_loss / max(1, len(valid_loader))
        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_dict, save_path)
            print(f"Saved Best Model to {save_path}")

    print("Correlation Pretraining Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/root/autodl-fs')
    parser.add_argument('--dataset_name', type=str, default='mosi')
    parser.add_argument('--margin', type=float, default=0.2)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--model_save_name', type=str, default='correlation_model')
    parser.add_argument('--perturbation_ratio', type=float, default=0.0)
    parser.add_argument('--max_samples', type=int, default=None)

    # 你原来 get_collate_fn 里 max_len 写死 400，我这里给成可调
    parser.add_argument('--max_len', type=int, default=400)

    args = parser.parse_args()
    train(args)
