import torch
from torch import nn
import sys
import os
from src import models
from src.utils import *
import torch.optim as optim
import numpy as np
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.eval_metrics import *
# 引入 sklearn 指标库
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from modality_correlation.correlation_loss import TripleLoss

def initiate(hyp_params, train_loader, valid_loader, test_loader):
    model = getattr(models, hyp_params.model+'Model')(hyp_params)

    if hyp_params.use_cuda:
        model = model.cuda()

    optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
    criterion = getattr(nn, hyp_params.criterion)()
    
    # 初始化对比损失函数
    contrastive_criterion = TripleLoss(margin=hyp_params.margin if hasattr(hyp_params, 'margin') else 0.2)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1, verbose=True)
    
    settings = {'model': model,
                'optimizer': optimizer,
                'criterion': criterion,
                'contrastive_criterion': contrastive_criterion,
                'scheduler': scheduler}
    
    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader)


def compute_metrics(results, truths):
    """
    辅助函数：计算核心评估指标 (MAE, Acc7, Acc2, F1)
    """
    test_preds = results.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()

    # 1. 计算 MAE
    mae = mean_absolute_error(test_truth, test_preds)

    # 2. 排除 0 值计算二分类指标 (CMU-MOSEI 标准做法)
    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])
    
    if len(non_zeros) > 0:
        binary_truth = (test_truth[non_zeros] > 0)
        binary_preds = (test_preds[non_zeros] > 0)
        acc2 = accuracy_score(binary_truth, binary_preds)
        f1 = f1_score(binary_truth, binary_preds, average='weighted')
    else:
        acc2 = 0.0
        f1 = 0.0

    # 3. 计算 Acc-7
    preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
    truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
    preds_a7 = np.round(preds_a7)
    truth_a7 = np.round(truth_a7)
    acc7 = np.sum(preds_a7 == truth_a7) / float(len(truth_a7))

    return mae, acc7, acc2, f1


def train_model(settings, hyp_params, train_loader, valid_loader, test_loader):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']
    contrastive_criterion = settings['contrastive_criterion']
    scheduler = settings['scheduler']
    
    beta = hyp_params.beta if hasattr(hyp_params, 'beta') else 0.1 

    def train(model, optimizer, criterion, contrastive_criterion, epoch):
        epoch_loss = 0
        model.train()
        num_batches = hyp_params.n_train // hyp_params.batch_size
        proc_loss, proc_size = 0, 0
        start_time = time.time()
        
        for i_batch, batch_data in enumerate(train_loader):
            # === 数据解包 (保持你的鲁棒逻辑) ===
            try:
                if len(batch_data) == 8:
                    metas, text, audio, vision, text_neg, audio_neg, vision_neg, eval_attr = batch_data
                elif len(batch_data) == 2:
                    inputs, eval_attr = batch_data
                    if len(inputs) == 7:
                        metas, text, audio, vision, text_neg, audio_neg, vision_neg = inputs
                    else:
                        raise ValueError("Batch input size unexpected.")
                else:
                    raise ValueError("Unknown batch structure.")
            except ValueError:
                 print("Error: DataLoader must provide negative samples for End-to-End training!")
                 sys.exit(1)

            model.zero_grad()
                
            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    text, audio, vision = text.cuda(), audio.cuda(), vision.cuda()
                    text_neg, audio_neg, vision_neg = text_neg.cuda(), audio_neg.cuda(), vision_neg.cuda()
                    eval_attr = eval_attr.cuda()
                    if hyp_params.dataset == 'iemocap':
                        eval_attr = eval_attr.long()
            
            batch_size = text.size(0)
            
            # === 前向传播 ===
            preds, _, seq_features = model(text, audio, vision)
            F_T, F_A, F_V = seq_features

            if isinstance(model, nn.DataParallel):
                corr_module = model.module.corr_model
            else:
                corr_module = model.corr_model
            
            F_T_n, F_A_n, F_V_n = corr_module(text_neg, audio_neg, vision_neg)

            # === 计算 Loss ===
            if hyp_params.dataset == 'iemocap':
                preds = preds.view(-1, 2)
                eval_attr = eval_attr.view(-1)
            else:
                preds = preds.view(-1, hyp_params.output_dim)
                eval_attr = eval_attr.view(-1, hyp_params.output_dim)
            
            task_loss = criterion(preds, eval_attr)

            loss_A = contrastive_criterion(F_A, F_T, F_T_n) + contrastive_criterion(F_A, F_V, F_V_n)
            loss_T = contrastive_criterion(F_T, F_A, F_A_n) + contrastive_criterion(F_T, F_V, F_V_n)
            loss_V = contrastive_criterion(F_V, F_A, F_A_n) + contrastive_criterion(F_V, F_T, F_T_n)
            
            contrastive_loss = (loss_A + loss_T + loss_V) / 3.0

            combined_loss = task_loss + beta * contrastive_loss
            
            combined_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()
            
            proc_loss += combined_loss.item() * batch_size
            proc_size += batch_size
            epoch_loss += combined_loss.item() * batch_size
            
            # === 详细的 Batch 日志 ===
            if i_batch % hyp_params.log_interval == 0 and i_batch > 0:
                avg_loss = proc_loss / proc_size
                elapsed_time = time.time() - start_time
                print('Epoch {:2d} | Batch {:3d}/{:3d} | Time {:4.0f}ms | Total {:.4f} (Task: {:.4f} | Cont: {:.4f})'.
                      format(epoch, i_batch, num_batches, elapsed_time * 1000 / hyp_params.log_interval, 
                             avg_loss, task_loss.item(), contrastive_loss.item()))
                proc_loss, proc_size = 0, 0
                start_time = time.time()
                
        return epoch_loss / hyp_params.n_train

    def evaluate(model, criterion, test=False):
        model.eval()
        loader = test_loader if test else valid_loader
        total_loss = 0.0
        
        results = []
        truths = []

        with torch.no_grad():
            for i_batch, batch_data in enumerate(loader):
                if len(batch_data) == 8:
                    metas, text, audio, vision, _, _, _, eval_attr = batch_data
                elif len(batch_data) == 2:
                     inputs, eval_attr = batch_data
                     if len(inputs) == 7:
                        metas, text, audio, vision, _, _, _ = inputs
                     else:
                        metas, text, audio, vision = inputs
                
                if hyp_params.use_cuda:
                    with torch.cuda.device(0):
                        text, audio, vision, eval_attr = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda()
                        if hyp_params.dataset == 'iemocap':
                            eval_attr = eval_attr.long()
                
                batch_size = text.size(0)
                preds, _, _ = model(text, audio, vision)
                
                if hyp_params.dataset == 'iemocap':
                    preds = preds.view(-1, 2)
                    eval_attr = eval_attr.view(-1)
                else:
                    preds = preds.view(-1, hyp_params.output_dim)
                    eval_attr = eval_attr.view(-1, hyp_params.output_dim)

                if hyp_params.criterion == 'CrossEntropyLoss':
                    total_loss += criterion(preds, eval_attr.long()).item() * batch_size
                else:
                    total_loss += criterion(preds, eval_attr).item() * batch_size

                results.append(preds)
                truths.append(eval_attr)

        avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)

        results = torch.cat(results)
        truths = torch.cat(truths)
        return avg_loss, results, truths

    best_valid = -1
    
    # === 打印漂亮的表头 ===
    print("\n" + "="*95)
    print(f"{'Epoch':^6} | {'Set':^5} | {'Loss':^8} | {'MAE':^8} | {'Acc-7':^8} | {'Acc-2':^8} | {'F1':^8} | {'Time':^8}")
    print("="*95)

    for epoch in range(1, hyp_params.num_epochs+1):
        start = time.time()
        
        # 1. 训练
        train(model, optimizer, criterion, contrastive_criterion, epoch)
        
        # 2. 评估
        val_loss, val_results, val_truths = evaluate(model, criterion, test=False)
        test_loss, test_results, test_truths = evaluate(model, criterion, test=True)
        
        # 3. 计算所有指标
        v_mae, v_acc7, v_acc2, v_f1 = compute_metrics(val_results, val_truths)
        t_mae, t_acc7, t_acc2, t_f1 = compute_metrics(test_results, test_truths)

        end = time.time()
        duration = end-start
        scheduler.step(val_loss)

        # === 打印 Epoch 表格行 ===
        print("-" * 95)
        print(f"{epoch:^6} | {'Valid':^5} | {val_loss:.4f}   | {v_mae:.4f}   | {v_acc7:.4f}   | {v_acc2:.4f}   | {v_f1:.4f}   | {duration:.2f}s")
        print(f"{'':^6} | {'Test':^5}  | {test_loss:.4f}   | {t_mae:.4f}   | {t_acc7:.4f}   | {t_acc2:.4f}   | {t_f1:.4f}   |")
        print("-" * 95)
        
        # if val_loss < best_valid:
        #     print(f"Saved best model! (Val Loss: {val_loss:.4f})")
        #     os.makedirs('pre_trained_models', exist_ok=True)
        #     torch.save(model.state_dict(), f'pre_trained_models/{hyp_params.name}.pt')
        #     best_valid = val_loss
        if v_f1 > best_valid:  # 如果当前 F1 分数比历史最高还高
            print(f"Saved best model! (Val F1: {v_f1:.4f} | Acc: {v_acc2:.4f})")
            os.makedirs('pre_trained_models', exist_ok=True)
            torch.save(model.state_dict(), f'pre_trained_models/{hyp_params.name}.pt')
            best_valid = v_f1

    # === 最终评估 ===
    print(f"\nTraining finished. Loading best model from pre_trained_models/{hyp_params.name}.pt...")
    model.load_state_dict(torch.load(f'pre_trained_models/{hyp_params.name}.pt'))
    
    print("Final Evaluation on Test Set:")
    _, results, truths = evaluate(model, criterion, test=True)
    
    if hyp_params.dataset == "mosei_senti":
        eval_mosei_senti(results, truths, True)
    elif hyp_params.dataset == 'ch_sims':
        eval_ch_sims(results, truths, True)

    sys.stdout.flush()
    return best_valid