import torch
import argparse
from src.utils import *
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import random
import numpy as np
from src import train
from datetime import datetime
import gpustat
from modality_correlation.correlation_dataset import UnifiedMultimodalDataset

print("before main")
print(gpustat.print_gpustat())

parser = argparse.ArgumentParser(description='MOSEI Sentiment Analysis')
parser.add_argument('-f', default='', type=str)

# Fixed
parser.add_argument('--model', type=str, default='MulT', help='name of the model to use (Transformer, etc.)')

# Tasks
parser.add_argument('--vonly', action='store_true', help='use the crossmodal fusion into v (default: False)')
parser.add_argument('--aonly', action='store_true', help='use the crossmodal fusion into a (default: False)')
parser.add_argument('--lonly', action='store_true', help='use the crossmodal fusion into l (default: False)')
parser.add_argument('--aligned', action='store_true', help='consider aligned experiment or not (default: False)')
parser.add_argument('--dataset', type=str, default='mosei_senti', help='dataset to use (default: mosei_senti)')
parser.add_argument('--data_path', type=str, default='data', help='path for storing the dataset')

# Dropouts
parser.add_argument('--attn_dropout', type=float, default=0.1, help='attention dropout')
parser.add_argument('--attn_dropout_a', type=float, default=0.1, help='attention dropout (for audio)')
parser.add_argument('--attn_dropout_v', type=float, default=0.1, help='attention dropout (for visual)')
parser.add_argument('--relu_dropout', type=float, default=0.1, help='relu dropout')
parser.add_argument('--embed_dropout', type=float, default=0.25, help='embedding dropout')
parser.add_argument('--res_dropout', type=float, default=0.1, help='residual block dropout')
parser.add_argument('--out_dropout', type=float, default=0.0, help='output layer dropout')

# Architecture
parser.add_argument('--nlevels', type=int, default=2, help='number of layers in the network (default: 5)')
parser.add_argument('--num_heads', type=int, default=2, help='number of heads for the transformer network (default: 5)')
parser.add_argument('--attn_mask', action='store_false', help='use attention mask for Transformer (default: true)')

# Tuning
parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='batch size (default: 24)')
parser.add_argument('--clip', type=float, default=0.8, help='gradient clip value (default: 0.8)')
parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate (default: 3 * 1e-4)')
parser.add_argument('--optim', type=str, default='Adam',help='optimizer to use (default: Adam)')
parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs (default: 40)')
parser.add_argument('--when', type=int, default=10, help='when to decay learning rate (default: 20)')
parser.add_argument('--batch_chunk', type=int, default=1, help='number of chunks per batch (default: 1)')

# Logistics
parser.add_argument('--log_interval', type=int, default=30, help='frequency of result logging (default: 30)')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--no_cuda', action='store_true', help='do not use cuda')
parser.add_argument('--name', type=str, default='mult', help='name of the trial (default: "mult")')

# Disturbance control:
parser.add_argument('--perturbation_ratio', type=float, default=0.0, help='Proportion of perturbed samples used in the training set')
parser.add_argument('--sample_ratio', type=float, default=1.0, help='Proportion of data retained in the training set')
parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to use')
# 在 Tuning 部分添加
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay (default: 1e-4)')
# ======================================================
# 新增: 端到端训练需要的超参数
# ======================================================
parser.add_argument('--beta', type=float, default=0.1, help='Weight for contrastive loss in total loss')
parser.add_argument('--margin', type=float, default=0.2, help='Margin for TripleLoss')
parser.add_argument('--corr_model_path', type=str, default='',
                    help='Optional: path to pretrained correlation model (state_dict). Empty => train from scratch.')
parser.add_argument('--freeze_corr_model', action='store_true',
                    help='Freeze correlation model parameters (no gradient updates).')
parser.add_argument('--corr_bias_grad', action=argparse.BooleanOptionalAction, default=True,
                    help='Allow task-loss gradient to flow into corr_model through correlation bias (higher memory).')
# 在 main.py 的 argparse 部分加入这行
parser.add_argument('--use_correlation', action=argparse.BooleanOptionalAction, default=True,
                    help='Enable correlation module & (optionally) contrastive loss. Use --no-use_correlation to disable.')
args = parser.parse_args()
# args.data_path = "/root/CH-SIMS"
# args.dataset = "ch_sims"

torch.manual_seed(args.seed)
dataset = str.lower(args.dataset.strip())
valid_partial_mode = args.lonly + args.vonly + args.aonly

if valid_partial_mode == 0:
    args.lonly = args.vonly = args.aonly = True
elif valid_partial_mode != 1:
    raise ValueError("You can only choose one of {l/v/a}only.")

use_cuda = False
output_dim_dict = {
    'mosei_senti': 1,
    'ch_sims': 1,
}
criterion_dict = {}

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    if args.no_cuda:
        print("WARNING: You have a CUDA device, so you should probably not run with --no_cuda")
        use_cuda = False
    else:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        use_cuda = True
else:
    use_cuda = False
    torch.manual_seed(args.seed)

####################################################################
#
# Load the dataset (aligned or non-aligned)
#
####################################################################
print("before loading the data")
print(gpustat.print_gpustat())
print("Start loading the data....")

# If beta>0, we need negative samples for contrastive loss during training.
train_for_correlation = bool(args.use_correlation) and float(args.beta) > 0.0


# ======================================================
# 单阶段(E2E)训练：train split 可选返回负样本，用于对比/三元组损失 
# ======================================================
train_data = UnifiedMultimodalDataset(
    dataset_path=args.data_path,
    data=args.dataset,
    split_type='train',
    if_align=args.aligned,
    max_samples=args.max_samples,
    for_correlation=train_for_correlation, 
    perturbation_ratio=0,
    noise_std=0.05
)

print("train data loaded")
print(gpustat.print_gpustat())

valid_data = UnifiedMultimodalDataset(
    dataset_path=args.data_path,
    data=args.dataset,
    split_type='valid',
    if_align=args.aligned,
    max_samples=args.max_samples,
    for_correlation=False, # 关键修改
    perturbation_ratio=0,
    strategy_weights=[1/3, 1/3, 1/3],
    noise_std=0.05
)

print("valid data loaded")
print(gpustat.print_gpustat())

test_data = UnifiedMultimodalDataset(
    dataset_path=args.data_path,
    data=args.dataset,
    split_type='test',
    if_align=args.aligned,
    max_samples=args.max_samples,
    for_correlation=False, # 关键修改
    perturbation_ratio=0,
    strategy_weights=[1/3, 1/3, 1/3],
    noise_std=0.05
)

print("test data loaded")
print(gpustat.print_gpustat())

# main.py 修改建议

def get_collate_fn(hyp_params, include_neg: bool = False):
    def collate_fn(batch):
        """
        Collate function (supports both modes):
        - include_neg=False (Stage2/纯任务): returns (metas, text, audio, vision, label)
        - include_neg=True  (E2E/单阶段):  returns (metas, text, audio, vision, text_neg, audio_neg, vision_neg, label)
        """
        max_text_len = hyp_params.l_len
        max_audio_len = hyp_params.a_len
        max_vision_len = hyp_params.v_len

        metas = []
        labels = []

        texts = []
        audios = []
        visions = []

        texts_neg = []
        audios_neg = []
        visions_neg = []

        for item in batch:
            # Regular mode: ((meta, text, audio, vision), label, (meta,))
            # Correlation mode: ((meta, text_pos, audio_pos, vision_pos), (text_neg, audio_neg, vision_neg), label, META)
            if include_neg:
                (meta, t_pos, a_pos, v_pos), (t_neg, a_neg, v_neg), label, _META = item
                metas.append(meta)
                labels.append(label)

                texts.append(t_pos[:max_text_len])
                audios.append(a_pos[:max_audio_len])
                visions.append(v_pos[:max_vision_len])

                texts_neg.append(t_neg[:max_text_len])
                audios_neg.append(a_neg[:max_audio_len])
                visions_neg.append(v_neg[:max_vision_len])
            else:
                (meta, t, a, v), label, _ = item
                metas.append(meta)
                labels.append(label)

                texts.append(t[:max_text_len])
                audios.append(a[:max_audio_len])
                visions.append(v[:max_vision_len])

        texts_padded = pad_sequence(texts, batch_first=True)
        audios_padded = pad_sequence(audios, batch_first=True)
        visions_padded = pad_sequence(visions, batch_first=True)
        labels_tensor = torch.stack(labels).view(-1)

        if not include_neg:
            return (metas, texts_padded, audios_padded, visions_padded, labels_tensor)

        texts_neg_padded = pad_sequence(texts_neg, batch_first=True)
        audios_neg_padded = pad_sequence(audios_neg, batch_first=True)
        visions_neg_padded = pad_sequence(visions_neg, batch_first=True)

        return (metas,
                texts_padded, audios_padded, visions_padded,
                texts_neg_padded, audios_neg_padded, visions_neg_padded,
                labels_tensor)

    return collate_fn

print('Finish loading the data....')
if not args.aligned:
    print("### Note: You are running in unaligned mode.")

####################################################################
#
# Hyperparameters
#
####################################################################

hyp_params = args
hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v = train_data.get_dim()
hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = train_data.get_seq_len()
hyp_params.layers = args.nlevels
hyp_params.use_cuda = use_cuda
hyp_params.dataset = dataset
hyp_params.when = args.when
hyp_params.batch_chunk = args.batch_chunk
hyp_params.n_train, hyp_params.n_valid, hyp_params.n_test = len(train_data), len(valid_data), len(test_data)
hyp_params.model = str.upper(args.model.strip())
hyp_params.output_dim = output_dim_dict.get(dataset, 1)
hyp_params.criterion = criterion_dict.get(dataset, 'L1Loss')
hyp_params.criterion = 'MSELoss'
# hyp_params.criterion = 'L1Loss'


# correlation model init (optional pretrained)
if hasattr(args, 'corr_model_path') and args.corr_model_path:
    hyp_params.corr_model_path = args.corr_model_path

# newly added
if args.aligned:
    predefined_max_len = 100
else:
    predefined_max_len = 300  # 给 Unaligned 足够空间
hyp_params.l_len = min(hyp_params.l_len, predefined_max_len)
hyp_params.a_len = min(hyp_params.a_len, predefined_max_len)
hyp_params.v_len = min(hyp_params.v_len, predefined_max_len)

current_time = datetime.now().strftime("%Y%m%d_%H")
hyp_params.name = ("DyCoTri" if hyp_params.use_correlation else "MulT") + "_" + current_time

# ======================================================
# 修改 3: 确保所有 Loader 使用正确的 Collate Fn
# ======================================================
print("before train_loader")
print(gpustat.print_gpustat())
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                          collate_fn=get_collate_fn(hyp_params, include_neg=train_for_correlation))

print("before valid_loader")
print(gpustat.print_gpustat())
valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False,
                          collate_fn=get_collate_fn(hyp_params, include_neg=False))

print("before test_loader")
print(gpustat.print_gpustat())
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                          collate_fn=get_collate_fn(hyp_params, include_neg=False))


if __name__ == '__main__':
    test_loss = train.initiate(hyp_params, train_loader, valid_loader, test_loader)