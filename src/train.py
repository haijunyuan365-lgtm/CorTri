import os
import sys
import time
import numpy as np

import torch
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src import models
from src.utils import *
from src.eval_metrics import *

from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from modality_correlation.correlation_loss import TripleLoss


# =========================
# Helpers
# =========================
def _to_numpy(x: torch.Tensor):
    return x.view(-1).detach().cpu().numpy()


def _is_finite_tensor(x: torch.Tensor) -> bool:
    if x is None:
        return True
    if not torch.is_tensor(x):
        return True
    return torch.isfinite(x).all().item()


def _nan_to_num_(x: torch.Tensor, nan=0.0, posinf=0.0, neginf=0.0):
    # in-place safe cast
    return torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)


def _print_tensor_stats(name: str, x: torch.Tensor, max_items: int = 5):
    if x is None or (not torch.is_tensor(x)):
        return
    with torch.no_grad():
        finite = torch.isfinite(x)
        n_total = x.numel()
        n_bad = n_total - int(finite.sum().item())
        if n_bad > 0:
            bad_vals = x[~finite].flatten()
            sample = bad_vals[:max_items].detach().cpu().numpy()
            print(f"[NAN-TRACE] {name}: non-finite {n_bad}/{n_total}, sample={sample}")
        else:
            # optional: print very large values
            max_abs = float(x.abs().max().item()) if n_total > 0 else 0.0
            if max_abs > 1e4:
                print(f"[TRACE] {name}: max_abs={max_abs:.3e}")


def initiate(hyp_params, train_loader, valid_loader, test_loader):
    model = getattr(models, hyp_params.model + 'Model')(hyp_params)

    if hyp_params.use_cuda:
        model = model.cuda()

    # -------- optimizer: optionally separate corr_model with smaller lr --------
    lr = hyp_params.lr
    wd = hyp_params.weight_decay
    optim_name = hyp_params.optim

    beta = hyp_params.beta if hasattr(hyp_params, 'beta') else 0.0
    use_corr = bool(getattr(hyp_params, 'use_correlation', False))

    # Default: corr_model lr multiplier (stability for end-to-end)
    corr_lr_mult = float(getattr(hyp_params, 'corr_lr_mult', 0.1))
    separate_corr_lr = bool(getattr(hyp_params, 'separate_corr_lr', True))

    params = model.parameters()
    optimizer = None

    if use_corr and beta > 0 and separate_corr_lr and hasattr(model, 'corr_model'):
        # Split param groups: corr_model smaller lr, others normal lr
        corr_params = list(model.corr_model.parameters())
        corr_param_ids = set(id(p) for p in corr_params)

        main_params = [p for p in model.parameters() if id(p) not in corr_param_ids]

        optimizer = getattr(optim, optim_name)(
            [
                {"params": main_params, "lr": lr, "weight_decay": wd},
                {"params": corr_params, "lr": lr * corr_lr_mult, "weight_decay": wd},
            ]
        )
        print(f"[OPT] Using param groups: main lr={lr}, corr lr={lr * corr_lr_mult} (mult={corr_lr_mult})")
    else:
        optimizer = getattr(optim, optim_name)(
            model.parameters(),
            lr=lr,
            weight_decay=wd
        )
        print(f"[OPT] Using single param group lr={lr}")

    criterion = getattr(nn, hyp_params.criterion)()

    # contrastive criterion (for compatibility; only used when beta>0 and neg exists)
    contrastive_criterion = TripleLoss(margin=hyp_params.margin if hasattr(hyp_params, 'margin') else 0.2)

    scheduler = ReduceLROnPlateau(
        optimizer, mode='min',
        patience=hyp_params.when, factor=0.1, verbose=True
    )

    settings = {
        'model': model,
        'optimizer': optimizer,
        'criterion': criterion,
        'contrastive_criterion': contrastive_criterion,
        'scheduler': scheduler,
        'best_valid': -1.0
    }

    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader)


def compute_metrics(results, truths):
    """
    计算 MAE, Acc7, Acc2, F1
    """
    # 防止 sklearn 因 NaN/Inf 崩溃
    results = _nan_to_num_(results, nan=0.0, posinf=0.0, neginf=0.0)
    truths = _nan_to_num_(truths, nan=0.0, posinf=0.0, neginf=0.0)

    test_preds = _to_numpy(results)
    test_truth = _to_numpy(truths)

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
    acc7 = float(np.sum(preds_a7 == truth_a7)) / float(len(truth_a7)) if len(truth_a7) > 0 else 0.0

    return mae, acc7, acc2, f1


def train_model(settings, hyp_params, train_loader, valid_loader, test_loader):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']
    contrastive_criterion = settings['contrastive_criterion']
    scheduler = settings['scheduler']
    best_valid = settings.get('best_valid', -1.0)

    beta = hyp_params.beta if hasattr(hyp_params, 'beta') else 0.0  # 端到端时 >0
    use_corr = bool(getattr(hyp_params, 'use_correlation', False))

    # Optional: warmup beta to avoid sudden instability
    beta_warmup_epochs = int(getattr(hyp_params, 'beta_warmup_epochs', 0))  # e.g., 2
    beta_target = float(beta)

    # Optional: freeze corr_model for first N epochs (stability)
    freeze_corr_epochs = int(getattr(hyp_params, 'freeze_corr_epochs', 0))  # e.g., 2

    # Hard guard enabled
    nan_guard = bool(getattr(hyp_params, 'nan_guard', True))

    def _current_beta(epoch: int) -> float:
        if beta_target <= 0:
            return 0.0
        if beta_warmup_epochs <= 0:
            return beta_target
        if epoch <= beta_warmup_epochs:
            # linear warmup: 0 -> target
            return beta_target * float(epoch) / float(beta_warmup_epochs)
        return beta_target

    def _maybe_freeze_corr(epoch: int):
        if not hasattr(model, 'corr_model'):
            return
        if freeze_corr_epochs <= 0:
            return
        req_grad = not (epoch <= freeze_corr_epochs)
        for p in model.corr_model.parameters():
            p.requires_grad = req_grad
        if epoch == 1:
            print(f"[FREEZE] corr_model frozen for first {freeze_corr_epochs} epochs")
        if epoch == freeze_corr_epochs + 1:
            print("[FREEZE] corr_model unfrozen")

    def train_one_epoch(epoch: int):
        epoch_loss = 0.0
        model.train()
        _maybe_freeze_corr(epoch)

        num_batches = len(train_loader)
        proc_loss, proc_size = 0.0, 0
        start_time = time.time()

        cur_beta = _current_beta(epoch)

        for i_batch, batch_data in enumerate(train_loader):
            # --------------------------
            # 1) Unpack batch
            # --------------------------
            text_neg, audio_neg, vision_neg = None, None, None
            try:
                if len(batch_data) == 5:
                    metas, text, audio, vision, eval_attr = batch_data
                elif len(batch_data) >= 8:
                    metas, text, audio, vision, text_neg, audio_neg, vision_neg, eval_attr = batch_data[:8]
                elif len(batch_data) == 2:
                    inputs, eval_attr = batch_data
                    if len(inputs) == 7:
                        metas, text, audio, vision, text_neg, audio_neg, vision_neg = inputs
                    else:
                        raise ValueError("Batch input size unexpected.")
                else:
                    raise ValueError(f"Unknown batch structure with length {len(batch_data)}")
            except ValueError as e:
                print(f"Error in unpacking: {e}")
                sys.exit(1)

            # --------------------------
            # 2) Move to GPU
            # --------------------------
            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    text, audio, vision = text.cuda(), audio.cuda(), vision.cuda()
                    eval_attr = eval_attr.cuda()
                    if text_neg is not None:
                        text_neg, audio_neg, vision_neg = text_neg.cuda(), audio_neg.cuda(), vision_neg.cuda()
                    if hyp_params.dataset == 'iemocap':
                        eval_attr = eval_attr.long()

            # Input cleaning (防止数据里偶发 NaN)
            text = _nan_to_num_(text, nan=0.0, posinf=0.0, neginf=0.0)
            audio = _nan_to_num_(audio, nan=0.0, posinf=0.0, neginf=0.0)
            vision = _nan_to_num_(vision, nan=0.0, posinf=0.0, neginf=0.0)
            eval_attr = _nan_to_num_(eval_attr, nan=0.0, posinf=0.0, neginf=0.0)

            batch_size = text.size(0)

            # --------------------------
            # 3) Forward
            # --------------------------
            optimizer.zero_grad(set_to_none=True)

            outputs = model(text, audio, vision)
            preds = outputs[0]

            # Default contrastive loss = 0
            contrastive_loss = torch.zeros((), device=preds.device)

            # --------------------------
            # 4) Contrastive loss (only if enabled and neg exists)
            # --------------------------
            if use_corr and cur_beta > 0:
                if len(outputs) >= 3 and text_neg is not None:
                    _, _, seq_features = outputs
                    F_T, F_A, F_V = seq_features

                    # get corr_model
                    if isinstance(model, nn.DataParallel):
                        corr_module = model.module.corr_model
                    else:
                        corr_module = model.corr_model

                    # Forward neg through corr_model
                    F_T_n, F_A_n, F_V_n = corr_module(text_neg, audio_neg, vision_neg)

                    # Triplet losses
                    loss_A = contrastive_criterion(F_A, F_T, F_T_n) + contrastive_criterion(F_A, F_V, F_V_n)
                    loss_T = contrastive_criterion(F_T, F_A, F_A_n) + contrastive_criterion(F_T, F_V, F_V_n)
                    loss_V = contrastive_criterion(F_V, F_A, F_A_n) + contrastive_criterion(F_V, F_T, F_T_n)
                    contrastive_loss = (loss_A + loss_T + loss_V) / 3.0

            # --------------------------
            # 5) Task loss
            # --------------------------
            if hyp_params.dataset == 'iemocap':
                preds_ = preds.view(-1, 2)
                eval_ = eval_attr.view(-1)
            else:
                preds_ = preds.view(-1, hyp_params.output_dim)
                eval_ = eval_attr.view(-1, hyp_params.output_dim)

            task_loss = criterion(preds_, eval_)

            combined_loss = task_loss + cur_beta * contrastive_loss if cur_beta > 0 else task_loss

            # --------------------------
            # 6) NaN guard BEFORE backward
            # --------------------------
            if nan_guard:
                if (not _is_finite_tensor(task_loss)) or (cur_beta > 0 and not _is_finite_tensor(contrastive_loss)) or (not _is_finite_tensor(combined_loss)):
                    print(f"[NAN-GUARD] epoch={epoch} batch={i_batch} "
                          f"task={float(task_loss):.6f} cont={float(contrastive_loss):.6f} "
                          f"beta={cur_beta:.4f} total={float(combined_loss):.6f} -> SKIP STEP")

                    _print_tensor_stats("preds", preds)
                    if cur_beta > 0:
                        _print_tensor_stats("contrastive_loss", contrastive_loss)

                    optimizer.zero_grad(set_to_none=True)
                    continue

            # --------------------------
            # 7) Backward
            # --------------------------
            combined_loss.backward()

            # --------------------------
            # 8) NaN guard on gradients
            # --------------------------
            if nan_guard:
                bad_grad = False
                bad_name = None
                for n, p in model.named_parameters():
                    if p.grad is None:
                        continue
                    if not torch.isfinite(p.grad).all():
                        bad_grad = True
                        bad_name = n
                        break
                if bad_grad:
                    print(f"[NAN-GUARD] epoch={epoch} batch={i_batch} non-finite grad at {bad_name} -> SKIP STEP")
                    optimizer.zero_grad(set_to_none=True)
                    continue

            # --------------------------
            # 9) Step
            # --------------------------
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()

            # --------------------------
            # 10) Logging
            # --------------------------
            epoch_loss += float(combined_loss.item()) * batch_size
            proc_loss += float(combined_loss.item()) * batch_size
            proc_size += batch_size

            if i_batch % hyp_params.log_interval == 0 and i_batch > 0:
                avg_loss = proc_loss / max(proc_size, 1)
                elapsed_time = time.time() - start_time
                print('Epoch {:2d} | Batch {:3d}/{:3d} | Time {:4.0f}ms | Total {:.4f} (Task: {:.4f} | Cont: {:.4f})'.
                      format(epoch, i_batch, num_batches,
                             elapsed_time * 1000 / hyp_params.log_interval,
                             avg_loss, float(task_loss.item()), float(contrastive_loss.item())))
                proc_loss, proc_size = 0.0, 0
                start_time = time.time()

        denom = float(getattr(hyp_params, 'n_train', len(train_loader.dataset)))
        return epoch_loss / max(denom, 1.0)

    def evaluate(test=False):
        model.eval()
        loader = test_loader if test else valid_loader
        total_loss = 0.0

        results = []
        truths = []

        with torch.no_grad():
            for batch_data in loader:
                # Unpack (valid/test should be clean 5 elems, but keep compatibility)
                if len(batch_data) == 5:
                    metas, text, audio, vision, eval_attr = batch_data
                elif len(batch_data) >= 8:
                    metas, text, audio, vision, _, _, _, eval_attr = batch_data[:8]
                elif len(batch_data) == 2:
                    inputs, eval_attr = batch_data
                    if len(inputs) == 7:
                        metas, text, audio, vision, _, _, _ = inputs
                    else:
                        metas, text, audio, vision = inputs
                else:
                    print(f"[WARN] Unexpected batch structure len={len(batch_data)} in evaluate, skip batch")
                    continue

                if hyp_params.use_cuda:
                    with torch.cuda.device(0):
                        text, audio, vision, eval_attr = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda()
                        if hyp_params.dataset == 'iemocap':
                            eval_attr = eval_attr.long()

                # Input cleaning
                text = _nan_to_num_(text, nan=0.0, posinf=0.0, neginf=0.0)
                audio = _nan_to_num_(audio, nan=0.0, posinf=0.0, neginf=0.0)
                vision = _nan_to_num_(vision, nan=0.0, posinf=0.0, neginf=0.0)
                eval_attr = _nan_to_num_(eval_attr, nan=0.0, posinf=0.0, neginf=0.0)

                batch_size = text.size(0)

                outputs = model(text, audio, vision)
                preds = outputs[0]

                # reshape
                if hyp_params.dataset == 'iemocap':
                    preds_ = preds.view(-1, 2)
                    eval_ = eval_attr.view(-1)
                else:
                    preds_ = preds.view(-1, hyp_params.output_dim)
                    eval_ = eval_attr.view(-1, hyp_params.output_dim)

                # loss
                if hyp_params.criterion == 'CrossEntropyLoss':
                    loss_val = criterion(preds_, eval_.long())
                else:
                    loss_val = criterion(preds_, eval_)
                loss_val = _nan_to_num_(loss_val, nan=0.0, posinf=0.0, neginf=0.0)

                total_loss += float(loss_val.item()) * batch_size

                # 防止评估阶段把 NaN 传给 sklearn
                preds_ = _nan_to_num_(preds_, nan=0.0, posinf=0.0, neginf=0.0)
                eval_ = _nan_to_num_(eval_, nan=0.0, posinf=0.0, neginf=0.0)

                results.append(preds_)
                truths.append(eval_)

        denom = float(getattr(hyp_params, 'n_test', len(test_loader.dataset)) if test else getattr(hyp_params, 'n_valid', len(valid_loader.dataset)))
        avg_loss = total_loss / max(denom, 1.0)

        results = torch.cat(results, dim=0) if len(results) > 0 else torch.zeros((0,), device='cuda' if hyp_params.use_cuda else 'cpu')
        truths = torch.cat(truths, dim=0) if len(truths) > 0 else torch.zeros((0,), device='cuda' if hyp_params.use_cuda else 'cpu')
        return avg_loss, results, truths

    # =========================
    # Train Loop
    # =========================
    print("\n" + "=" * 95)
    print(f"{'Epoch':^6} | {'Set':^5} | {'Loss':^8} | {'MAE':^8} | {'Acc-7':^8} | {'Acc-2':^8} | {'F1':^8} | {'Time':^8}")
    print("=" * 95)

    for epoch in range(1, hyp_params.num_epochs + 1):
        start = time.time()

        train_one_epoch(epoch)

        val_loss, val_results, val_truths = evaluate(test=False)
        test_loss, test_results, test_truths = evaluate(test=True)

        v_mae, v_acc7, v_acc2, v_f1 = compute_metrics(val_results, val_truths)
        t_mae, t_acc7, t_acc2, t_f1 = compute_metrics(test_results, test_truths)

        duration = time.time() - start
        scheduler.step(val_loss)

        print("-" * 95)
        print(f"{epoch:^6} | {'Valid':^5} | {val_loss:.4f}   | {v_mae:.4f}   | {v_acc7:.4f}   | {v_acc2:.4f}   | {v_f1:.4f}   | {duration:.2f}s")
        print(f"{'':^6} | {'Test':^5}  | {test_loss:.4f}   | {t_mae:.4f}   | {t_acc7:.4f}   | {t_acc2:.4f}   | {t_f1:.4f}   |")
        print("-" * 95)

        if v_f1 > best_valid:
            print(f"Saved best model! (Val F1: {v_f1:.4f} | Acc: {v_acc2:.4f})")
            os.makedirs('pre_trained_models', exist_ok=True)
            torch.save(model.state_dict(), f'pre_trained_models/{hyp_params.name}.pt')
            best_valid = v_f1

    # =========================
    # Final Eval
    # =========================
    print(f"\nTraining finished. Loading best model from pre_trained_models/{hyp_params.name}.pt...")
    model.load_state_dict(torch.load(f'pre_trained_models/{hyp_params.name}.pt', map_location='cuda' if hyp_params.use_cuda else 'cpu'))

    print("Final Evaluation on Test Set:")
    _, results, truths = evaluate(test=True)

    if hyp_params.dataset == "mosei_senti":
        eval_mosei_senti(results, truths, True)
    elif hyp_params.dataset == 'ch_sims':
        eval_ch_sims(results, truths, True)

    sys.stdout.flush()
    return best_valid
