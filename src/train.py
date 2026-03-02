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
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from src.AR_loss import ARLoss

def initiate(hyp_params, train_loader, valid_loader, test_loader):
    model = getattr(models, hyp_params.model+'Model')(hyp_params)

    if hyp_params.use_cuda:
        model = model.cuda()

    optimizer = getattr(optim, hyp_params.optim)(
        model.parameters(), 
        lr=hyp_params.lr, 
        weight_decay=hyp_params.weight_decay
    )
    criterion = getattr(nn, hyp_params.criterion)()
    ar_criterion = ARLoss(reduction='mean')

    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1, verbose=True)
    
    settings = {'model': model,
                'optimizer': optimizer,
                'criterion': criterion,
                'ar_criterion': ar_criterion,
                'scheduler': scheduler,
                'best_valid': float('inf')}
    
    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader)


def compute_metrics(results, truths):
    test_preds = results.view(-1).cpu().detach().numpy()
    test_truth = truths.view(-1).cpu().detach().numpy()

    mae = mean_absolute_error(test_truth, test_preds)
    non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])
    
    if len(non_zeros) > 0:
        binary_truth = (test_truth[non_zeros] > 0)
        binary_preds = (test_preds[non_zeros] > 0)
        acc2 = accuracy_score(binary_truth, binary_preds)
        f1 = f1_score(binary_truth, binary_preds, average='weighted')
    else:
        acc2 = 0.0
        f1 = 0.0

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
    scheduler = settings['scheduler']
    ar_criterion = settings.get('ar_criterion', None)
    
    ar_weight = getattr(hyp_params, 'ar_weight', 0.0)  
    best_valid = settings.get('best_valid', float('inf'))
    
    def train(model, optimizer, criterion, epoch):
        epoch_loss = 0
        model.train()
        
        # 安全地尝试设置 corr_model 为 eval 模式
        if hasattr(model, 'corr_model') and model.corr_model is not None:
            model.corr_model.eval()

        num_batches = len(train_loader)
        proc_loss, proc_size = 0, 0
        start_time = time.time()
        
        for i_batch, batch_data in enumerate(train_loader):
            metas, text, audio, vision, eval_attr = batch_data

            optimizer.zero_grad(set_to_none=True)
                
            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    text, audio, vision = text.cuda(), audio.cuda(), vision.cuda()
                    eval_attr = eval_attr.cuda()
                    if hyp_params.dataset == 'iemocap':
                        eval_attr = eval_attr.long()
            
            batch_size = text.size(0)
            
            outputs = model(text, audio, vision)
            preds = outputs[0] 
            
            if hyp_params.dataset == 'iemocap':
                preds = preds.view(-1, 2)
                eval_attr = eval_attr.view(-1)
            else:
                preds = preds.view(-1, hyp_params.output_dim)
                eval_attr = eval_attr.view(-1, hyp_params.output_dim)
            
            task_loss = criterion(preds, eval_attr)
            
            ar_loss = torch.tensor(0.0, device=preds.device)
            if ar_criterion is not None and ar_weight > 0 and hyp_params.dataset != 'iemocap':
                ar_loss = ar_criterion(preds, eval_attr)

            combined_loss = task_loss + ar_weight * ar_loss
            
            combined_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()
            
            proc_loss += combined_loss.item() * batch_size
            proc_size += batch_size
            epoch_loss += combined_loss.item() * batch_size
            
            if i_batch % hyp_params.log_interval == 0 and i_batch > 0:
                avg_loss = proc_loss / proc_size
                elapsed_time = time.time() - start_time
                print('Epoch {:2d} | Batch {:3d}/{:3d} | Time {:4.0f}ms | Total {:.4f} (Task: {:.4f} | AR: {:.4f})'.
                    format(epoch, i_batch, num_batches, elapsed_time * 1000 / hyp_params.log_interval,
                            avg_loss, task_loss.item(), ar_loss.item()))
                proc_loss, proc_size = 0, 0
                start_time = time.time()
                
        return epoch_loss / hyp_params.n_train

    def log_bias_params(model):
        if not hasattr(model, "lambda_param"):
            return
        # 即使 use_correlation=False，lambda_param 可能还在，但没有意义
        # 这里只做简单的打印，不影响训练逻辑
        lam = torch.sigmoid(model.lambda_param.detach()).item()
        w1 = torch.softmax(torch.stack([model.w_tv, model.w_ta, model.w_va]), dim=0).detach().cpu().numpy()
        w2 = torch.softmax(torch.stack([model.w_tv, model.w_ta, model.w_av]), dim=0).detach().cpu().numpy()
        print(f"[DBG] lambda(sigmoid)={lam:.4f}  w_s1(tv,ta,va)={w1}  w_s2(tv,ta,av)={w2}")
        
    def evaluate(model, criterion, test=False):
        model.eval()
        loader = test_loader if test else valid_loader
        total_loss = 0.0
        
        results = []
        truths = []

        with torch.no_grad():
            for i_batch, batch_data in enumerate(loader):
                metas, text, audio, vision, eval_attr = batch_data
                
                if hyp_params.use_cuda:
                    with torch.cuda.device(0):
                        text, audio, vision, eval_attr = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda()
                        if hyp_params.dataset == 'iemocap':
                            eval_attr = eval_attr.long()
                
                batch_size = text.size(0)
                
                outputs = model(text, audio, vision)
                preds = outputs[0]
                
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

    
    print("\n" + "="*95)
    print(f"{'Epoch':^6} | {'Set':^5} | {'Loss':^8} | {'MAE':^8} | {'Acc-7':^8} | {'Acc-2':^8} | {'F1':^8} | {'Time':^8}")
    print("="*95)  
    for epoch in range(1, hyp_params.num_epochs+1):
        start = time.time()
        
        train(model, optimizer, criterion, epoch)
        
        val_loss, val_results, val_truths = evaluate(model, criterion, test=False)
        test_loss, test_results, test_truths = evaluate(model, criterion, test=True)
        
        v_mae, v_acc7, v_acc2, v_f1 = compute_metrics(val_results, val_truths)
        t_mae, t_acc7, t_acc2, t_f1 = compute_metrics(test_results, test_truths)

        end = time.time()
        duration = end-start
        scheduler.step(val_loss)

        print("-" * 95)
        print(f"{epoch:^6} | {'Valid':^5} | {val_loss:.4f}   | {v_mae:.4f}   | {v_acc7:.4f}   | {v_acc2:.4f}   | {v_f1:.4f}   | {duration:.2f}s")
        print(f"{'':^6} | {'Test':^5}  | {test_loss:.4f}   | {t_mae:.4f}   | {t_acc7:.4f}   | {t_acc2:.4f}   | {t_f1:.4f}   |")
        log_bias_params(model)
        print("-" * 95)
        
        if v_mae < best_valid:
            torch.save(model.state_dict(), f'pre_trained_models/{hyp_params.name}.pt')
            best_valid = v_mae
            print(f"Saved best model! (Val MAE: {v_mae:.4f} | F1: {v_f1:.4f} | Acc2: {v_acc2:.4f})")

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