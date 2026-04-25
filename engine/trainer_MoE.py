import os
import copy
import torch
import pandas as pd
import time
from engine.metrics import calculate_performance


class EarlyStopping:
    def __init__(self, patience=15, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
        self.val_loss_min = val_loss
        self.best_model_state = copy.deepcopy(model.state_dict())


def train_moe(train_loader, model, loss_fn, optimizer, args, valid_loader=None):
    """
    Train the MoE model for one fold.

    Returns
    -------
    model            : model with best-epoch weights restored
    best_val_metrics : dict of best validation metrics
    best_val_loss    : float
    train_times      : list of per-epoch train times
    val_times        : list of per-epoch val times
    epoch_history    : list of dicts — one row per epoch (for CSV/graph plotting)
                       Keys: epoch, train_main_loss, train_aux_loss,
                             val_main_loss, val_aux_loss, val_loss, + metric cols
    """
    early_stopping = EarlyStopping(patience=15, verbose=args.print_out)
    best_val_metrics = None
    best_val_loss = float('inf')

    train_times   = []
    val_times     = []
    epoch_history = []          # ← NEW: collect one row per epoch

    total_train_start = time.time()

    for epoch in range(args.epoch):
        model.train()
        epoch_loss      = 0.0
        epoch_main_loss = 0.0
        epoch_aux_loss  = 0.0
        start_time = time.time()

        num_batches = len(train_loader)
        for batch in train_loader:
            optimizer.zero_grad()
            output, aux_loss = model(batch)
            labels = batch['label'].to(output.device).float()

            main_loss  = loss_fn(output, labels)
            total_loss = main_loss + aux_loss

            total_loss.backward()
            optimizer.step()

            epoch_loss      += total_loss.item()
            epoch_main_loss += main_loss.item()
            epoch_aux_loss  += aux_loss.item()

        train_time = time.time() - start_time
        train_times.append(train_time)

        avg_main_loss = epoch_main_loss / num_batches
        avg_aux_loss  = epoch_aux_loss  / num_batches

        if args.print_out:
            print(f"\n--- Epoch {epoch} ---")
            print(f"Training Time: {train_time:.2f}s")
            print(f"Main Loss: {avg_main_loss:.4f} | Aux Loss: {avg_aux_loss:.4f}")

        if args.lr_decay and args.decay_interval and epoch % args.decay_interval == 0:
            optimizer.param_groups[0]['lr'] *= args.lr_decay

        # ── Per-epoch record (train side) ─────────────────────────────────────
        row = {
            'epoch':           epoch,
            'train_main_loss': avg_main_loss,
            'train_aux_loss':  avg_aux_loss,
        }

        if valid_loader:
            v_start = time.time()
            result_df, perf_df, v_loss = test_moe(valid_loader, model, loss_fn, args, split='Valid')
            val_time = time.time() - v_start
            val_times.append(val_time)

            val_metrics = perf_df.iloc[0].to_dict()

            if args.print_out:
                print(f"Validation Time: {val_time:.2f}s")
                print(f"Validation Loss: {v_loss:.4f}")
                if args.task == 'classification':
                    print(f"Classification Metrics: {val_metrics}")
                else:
                    print(f"Regression Metrics: {val_metrics}")

            # ── track best ───────────────────────────────────────────────────
            if v_loss < best_val_loss:
                best_val_loss    = v_loss
                best_val_metrics = val_metrics

            # Add val info to the row
            row['val_loss'] = v_loss
            for metric_name, metric_val in val_metrics.items():
                row[f'val_{metric_name}'] = metric_val

            early_stopping(v_loss, model)

        # FIX 1: Append row ONCE per epoch (after both train and optional val)
        epoch_history.append(row)

        if valid_loader and early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    total_train_time = time.time() - total_train_start
    print(f"\nTotal Training Time (Combined): {total_train_time:.2f}s")

    # Restore the best weights found during this fold before returning
    if early_stopping.best_model_state is not None:
        model.load_state_dict(early_stopping.best_model_state)

    return model, best_val_metrics, best_val_loss, train_times, val_times, epoch_history


def test_moe(loader, model, loss_fn, args, split='Test', cal_scores=None):
    from engine.conformal import apply_icp_reference_logic
    model.eval()
    preds, labels, com_ids, pro_ids = [], [], [], []
    icp_preds, icp_confs, icp_lows, icp_highs = [], [], [], []
    total_loss      = 0.0
    total_main_loss = 0.0
    total_aux_loss  = 0.0
    
    # Pre-calculate quantile for regression for speed
    if args.task == 'regression' and cal_scores is not None:
        import numpy as np
        q_val = np.quantile(cal_scores, 1.0 - args.confidence)
    else:
        q_val = None

    with torch.no_grad():
        for batch in loader:
            output, aux_loss = model(batch)
            batch_labels = batch['label'].to(output.device).float()
            main_loss = loss_fn(output, batch_labels)

            loss             = main_loss + aux_loss
            total_loss      += loss.item()
            total_main_loss += main_loss.item()
            total_aux_loss  += aux_loss.item()

            if args.task == 'classification':
                probs = torch.sigmoid(output).cpu().numpy().tolist()
                preds.extend(probs)
                if cal_scores is not None:
                    # Apply reference ICP logic (Selective Prediction)
                    icp_res = apply_icp_reference_logic(output, cal_scores, args.task, alpha=1.0 - args.confidence)
                    icp_preds.extend([res[0] for res in icp_res])
                    icp_confs.extend([res[1] for res in icp_res])
            else:
                out_np = output.cpu().numpy().tolist()
                preds.extend(out_np)
                if cal_scores is not None:
                    # Apply proper ICP for regression using pre-calculated q_val
                    icp_res = apply_icp_reference_logic(output, cal_scores, args.task, 
                                                        alpha=1.0 - args.confidence, q=q_val)
                    icp_lows.extend([res[1] for res in icp_res])
                    icp_highs.extend([res[2] for res in icp_res])

            labels.extend(batch_labels.cpu().numpy().tolist())
            com_ids.extend(batch['com_id'])
            pro_ids.extend(batch['pro_id'])

    res_dict = {
        'com_id':    com_ids,
        'pro_id':    pro_ids,
        'pred':      preds,
        args.label:  labels
    }
    if cal_scores is not None:
        if args.task == 'classification':
            res_dict['predByICP'] = icp_preds
            res_dict['confICP']   = icp_confs
        else:
            res_dict['icp_low']  = icp_lows
            res_dict['icp_high'] = icp_highs

    result_df = pd.DataFrame(res_dict)

    metrics  = calculate_performance(result_df, args)
    avg_loss = total_loss / len(loader)

    if args.print_out:
        print(f"[{split}] Main Loss: {total_main_loss/len(loader):.4f} | "
              f"Aux Loss: {total_aux_loss/len(loader):.4f}")

    return result_df, pd.DataFrame([metrics], columns=args.metrics), avg_loss
