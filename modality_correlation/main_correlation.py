import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import argparse
import sys
import os
import time
from tqdm import tqdm

# 添加项目根目录
sys.path.append(os.getcwd())

from modality_correlation.correlation_dataset import UnifiedMultimodalDataset
from modality_correlation.correlation_models import CorrelationModel
from modality_correlation.correlation_loss import TripleLoss

# CorMulT 的默认配置
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

# CorMulT 使用的 collate_fn (不做复杂处理，只是 padding)
def get_collate_fn(max_len=400): # Unaligned 建议设为 400 或 500
    def collate_fn(batch):
        # batch item: ((meta, text, audio, vision), (text_neg, ...), label, ...)
        
        # 1. 截断 (避免显存爆炸)
        texts = [item[0][1][:max_len] for item in batch]
        audios = [item[0][2][:max_len] for item in batch]
        visions = [item[0][3][:max_len] for item in batch]
        
        texts_neg = [item[1][0][:max_len] for item in batch]
        audios_neg = [item[1][1][:max_len] for item in batch]
        visions_neg = [item[1][2][:max_len] for item in batch]
        
        # 2. Padding (batch_first=True)
        texts = pad_sequence(texts, batch_first=True)
        audios = pad_sequence(audios, batch_first=True)
        visions = pad_sequence(visions, batch_first=True)
        
        texts_neg = pad_sequence(texts_neg, batch_first=True)
        audios_neg = pad_sequence(audios_neg, batch_first=True)
        visions_neg = pad_sequence(visions_neg, batch_first=True)
        
        return texts, audios, visions, texts_neg, audios_neg, visions_neg

    return collate_fn

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Start Stage 1 Training for {args.dataset_name}...")
    
    # 1. 数据加载
    train_data = UnifiedMultimodalDataset(
        dataset_path=args.data_path,
        data=args.dataset_name,
        split_type='train',
        if_align=False, # Unaligned
        for_correlation=True, # 开启负样本生成
        perturbation_ratio=args.perturbation_ratio,
        noise_std=0.1
    )
    valid_data = UnifiedMultimodalDataset(
        dataset_path=args.data_path,
        data=args.dataset_name,
        split_type='valid',
        if_align=False,
        for_correlation=True,
        perturbation_ratio=0.0,
        noise_std=0.1
    )
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, 
                              collate_fn=get_collate_fn(), num_workers=4)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, 
                              collate_fn=get_collate_fn(), num_workers=4)

    # 2. 模型与优化器
    config = dataset_specific_configs[args.dataset_name]
    model = CorrelationModel(**config).to(device)
    
    # 如果有多张卡，CorMulT 使用了 DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = TripleLoss(margin=args.margin)

    # 3. 训练循环
    os.makedirs('pre_trained_models', exist_ok=True)
    save_path = f'pre_trained_models/{args.model_save_name}.pt'
    best_val_loss = float('inf')

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        total_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs}"):
            text, audio, vision, text_n, audio_n, vision_n = [x.to(device) for x in batch]
            
            optimizer.zero_grad()
            
            # Forward
            f_t, f_a, f_v = model(text, audio, vision)
            f_t_n, f_a_n, f_v_n = model(text_n, audio_n, vision_n)
            
            # Loss Calculation (Sum of all pairs)
            loss = 0
            loss += criterion(f_a, f_t, f_t_n) + criterion(f_a, f_v, f_v_n) # Anchor: Audio
            loss += criterion(f_t, f_a, f_a_n) + criterion(f_t, f_v, f_v_n) # Anchor: Text
            loss += criterion(f_v, f_a, f_a_n) + criterion(f_v, f_t, f_t_n) # Anchor: Vision
            
            loss = loss / 6.0
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                text, audio, vision, text_n, audio_n, vision_n = [x.to(device) for x in batch]
                f_t, f_a, f_v = model(text, audio, vision)
                f_t_n, f_a_n, f_v_n = model(text_n, audio_n, vision_n)
                
                loss = 0
                loss += criterion(f_a, f_t, f_t_n) + criterion(f_a, f_v, f_v_n)
                loss += criterion(f_t, f_a, f_a_n) + criterion(f_t, f_v, f_v_n)
                loss += criterion(f_v, f_a, f_a_n) + criterion(f_v, f_t, f_t_n)
                val_loss += (loss / 6.0).item()
        
        avg_val_loss = val_loss / len(valid_loader)
        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # 如果用了 DataParallel，保存 model.module
            state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state_dict, save_path)
            print(f"Saved Best Model to {save_path}")

    print("Correlation Pretraining Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/root/autodl-fs')
    parser.add_argument('--dataset_name', type=str, default='mosei_senti')
    parser.add_argument('--margin', type=float, default=0.2)
    parser.add_argument('--num_epochs', type=int, default=50)
    # CorMulT 默认配置: batch_size=24, lr=2e-4
    parser.add_argument('--batch_size', type=int, default=24) 
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--model_save_name', type=str, default='correlation_model')
    parser.add_argument('--perturbation_ratio', type=float, default=0.0) 
    parser.add_argument('--max_samples', type=int, default=None)
    
    args = parser.parse_args()
    train(args)