import torch
from torch import nn
import sys
from src import models
from src.utils import *
import torch.optim as optim
import numpy as np
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.eval_metrics import *
from modality_correlation.correlation_loss import TripleLoss # 导入修改后的细粒度矩阵 Loss

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
                'contrastive_criterion': contrastive_criterion, # 加入到 settings
                'scheduler': scheduler}
    
    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader)


def train_model(settings, hyp_params, train_loader, valid_loader, test_loader):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']
    contrastive_criterion = settings['contrastive_criterion']
    scheduler = settings['scheduler']
    
    # Beta 系数: 控制对比损失在总损失中的比重
    # 建议在 hyp_params 中添加 beta 参数，默认为 0.1 或 0.05
    beta = hyp_params.beta if hasattr(hyp_params, 'beta') else 0.1 

    def train(model, optimizer, criterion, contrastive_criterion):
        epoch_loss = 0
        model.train()
        num_batches = hyp_params.n_train // hyp_params.batch_size
        proc_loss, proc_size = 0, 0
        start_time = time.time()
        
        for i_batch, batch_data in enumerate(train_loader):
            # =========================================================
            # 修改 1: 适配包含负样本的数据解包
            # =========================================================
            # 假设 DataLoader 的 collate_fn 返回的是:
            # (metas, text, audio, vision, text_neg, audio_neg, vision_neg, labels)
            # 或者是 ((metas, t, a, v, tn, an, vn), labels)
            # 这里我们需要根据实际的 dataset 输出灵活处理。
            # 为了兼容 correlation_train.py 的逻辑，我们假设解包如下：
            
            try:
                if len(batch_data) == 8: # Flat tuple from correlation_train style
                    metas, text, audio, vision, text_neg, audio_neg, vision_neg, eval_attr = batch_data
                elif len(batch_data) == 2: # Standard (X, Y) style, assume X contains all
                    inputs, eval_attr = batch_data
                    if len(inputs) == 7:
                        metas, text, audio, vision, text_neg, audio_neg, vision_neg = inputs
                    else:
                        raise ValueError("Batch input size unexpected.")
                else:
                    raise ValueError("Unknown batch structure.")
            except ValueError:
                # Fallback: 如果数据加载器没提供负样本，可能无法进行对比学习训练
                # 这里为了稳健，如果解包失败，假设是普通的 (metas, t, a, v), label
                # 此时我们无法计算 contrastive loss (设为0)
                # 但这违背了端到端训练的设计，所以请务必确保 DataLoader 正确
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
            
            # =========================================================
            # 修改 2: 端到端前向传播 (Positive & Negative)
            # =========================================================
            
            # 1. 正样本前向传播 (Full Model)
            # output: [B, 1] 情感预测
            # seq_features: (F_T, F_A, F_V) 原始序列特征用于对比损失
            preds, _, seq_features = model(text, audio, vision)
            F_T, F_A, F_V = seq_features

            # 2. 负样本前向传播 (Correlation Model Only)
            # 只需要通过 MCE 模块获取特征，不需要跑后面的 Transformer，节省计算资源
            # 处理 DataParallel 的情况
            if isinstance(model, nn.DataParallel):
                corr_module = model.module.corr_model
            else:
                corr_module = model.corr_model
            
            # 获取负样本的序列特征
            F_T_n, F_A_n, F_V_n = corr_module(text_neg, audio_neg, vision_neg)

            # =========================================================
            # 修改 3: 计算联合损失 (Joint Loss)
            # =========================================================
            
            # Task Loss (情感预测)
            if hyp_params.dataset == 'iemocap':
                preds = preds.view(-1, 2)
                eval_attr = eval_attr.view(-1)
                task_loss = criterion(preds, eval_attr)
            else:
                preds = preds.view(-1, hyp_params.output_dim)
                eval_attr = eval_attr.view(-1, hyp_params.output_dim)
                task_loss = criterion(preds, eval_attr)

            # Contrastive Loss (物理相关性矩阵对比)
            # 计算两两模态间的 Triple Loss
            loss_A = contrastive_criterion(F_A, F_T, F_T_n) + contrastive_criterion(F_A, F_V, F_V_n)
            loss_T = contrastive_criterion(F_T, F_A, F_A_n) + contrastive_criterion(F_T, F_V, F_V_n)
            loss_V = contrastive_criterion(F_V, F_A, F_A_n) + contrastive_criterion(F_V, F_T, F_T_n)
            
            contrastive_loss = (loss_A + loss_T + loss_V) / 3.0

            # Total Loss
            combined_loss = task_loss + beta * contrastive_loss
            
            combined_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()
            
            proc_loss += combined_loss.item() * batch_size
            proc_size += batch_size
            epoch_loss += combined_loss.item() * batch_size
            
            if i_batch % hyp_params.log_interval == 0 and i_batch > 0:
                avg_loss = proc_loss / proc_size
                elapsed_time = time.time() - start_time
                print('Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Loss {:5.4f} (Task: {:.4f}, Cont: {:.4f})'.
                      format(epoch, i_batch, num_batches, elapsed_time * 1000 / hyp_params.log_interval, avg_loss, task_loss.item(), contrastive_loss.item()))
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
                # 评估阶段通常只需要正样本，但为了兼容 DataLoader 结构，我们依然需要正确解包
                if len(batch_data) == 8:
                    metas, text, audio, vision, _, _, _, eval_attr = batch_data
                elif len(batch_data) == 2:
                     inputs, eval_attr = batch_data
                     if len(inputs) == 7:
                        metas, text, audio, vision, _, _, _ = inputs
                     else:
                        # 兼容可能没有负样本的普通 Validation Loader
                        metas, text, audio, vision = inputs
                
                if hyp_params.use_cuda:
                    with torch.cuda.device(0):
                        text, audio, vision, eval_attr = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda()
                        if hyp_params.dataset == 'iemocap':
                            eval_attr = eval_attr.long()
                
                batch_size = text.size(0)
                
                # 评估时只需要 output
                preds, _, _ = model(text, audio, vision)
                
                if hyp_params.dataset == 'iemocap':
                    preds = preds.view(-1, 2)
                    eval_attr = eval_attr.view(-1)

                if hyp_params.criterion == 'CrossEntropyLoss':
                    total_loss += criterion(preds, eval_attr.long()).item() * batch_size
                else:
                    preds = preds.view(-1, hyp_params.output_dim)
                    eval_attr = eval_attr.view(-1, hyp_params.output_dim)
                    total_loss += criterion(preds, eval_attr).item() * batch_size

                results.append(preds)
                truths.append(eval_attr)

        avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)

        results = torch.cat(results)
        truths = torch.cat(truths)
        return avg_loss, results, truths

    best_valid = 1e8
    for epoch in range(1, hyp_params.num_epochs+1):
        start = time.time()
        
        # Train one epoch
        train(model, optimizer, criterion, contrastive_criterion)
        
        val_loss, _, _ = evaluate(model, criterion, test=False)
        test_loss, _, _ = evaluate(model, criterion, test=True)
        
        end = time.time()
        duration = end-start
        scheduler.step(val_loss)

        print("-"*50)
        print('Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(epoch, duration, val_loss, test_loss))
        print("-"*50)
        
        if val_loss < best_valid:
            print(f"Saved model at pre_trained_models/{hyp_params.name}.pt!")
            # Save the full model
            torch.save(model.state_dict(), f'pre_trained_models/{hyp_params.name}.pt')
            best_valid = val_loss

    # Load best model for final evaluation
    model.load_state_dict(torch.load(f'pre_trained_models/{hyp_params.name}.pt'))
    
    print("start evaluating...")
    _, results, truths = evaluate(model, criterion, test=True)

    if hyp_params.dataset == "mosei_senti":
        eval_mosei_senti(results, truths, True)
    elif hyp_params.dataset == 'ch_sims':
        eval_ch_sims(results, truths, True)

    sys.stdout.flush()
    # input('[Press Any Key to start another run]')
    return best_valid # Return metric