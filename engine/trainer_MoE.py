try:
    import wandb
except ImportError:
    wandb = None
import os
import torch
import pandas as pd
from tqdm import trange
from engine.utils import save_model
from engine.metrics import calculate_performance

def train_moe(train_loader, model, loss_fn, optimizer, args, valid_loader=None):
    best_perf = None
    for epoch in trange(args.epoch, desc="Training MoE"):
        model.train()
        epoch_loss = 0.0
        epoch_main_loss = 0.0
        epoch_aux_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            output, aux_loss = model(batch)
            labels = batch['label'].to(output.device).float()
            
            main_loss = loss_fn(output, labels)
            total_loss = main_loss + aux_loss
            
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            epoch_main_loss += main_loss.item()
            epoch_aux_loss += aux_loss.item()

        if args.print_out:
            print(f"Epoch {epoch} | Total Loss: {epoch_loss:.4f} | Main Loss: {epoch_main_loss:.4f} | Aux Loss: {epoch_aux_loss:.4f}")

        if wandb is not None and wandb.run is not None:
            wandb.log({
                'Train Total Loss': epoch_loss,
                'Train Main Loss': epoch_main_loss,
                'Train Aux Loss': epoch_aux_loss
            }, step=epoch)

        if args.lr_decay and args.decay_interval and epoch % args.decay_interval == 0:
            optimizer.param_groups[0]['lr'] *= args.lr_decay

        if valid_loader:
            result_df, perf_df = test_moe(valid_loader, model, loss_fn, args, split='Valid', epoch=epoch)
            if args.save_model:
                save_model(model, args, epoch, result_df=result_df)

    return model



def test_moe(loader, model, loss_fn, args, split='Test', epoch=0):
    model.eval()
    preds, labels, com_ids, pro_ids = [], [], [], []
    total_loss = 0.0
    total_main_loss = 0.0
    total_aux_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            output, aux_loss = model(batch)
            batch_labels = batch['label'].to(output.device).float()
            main_loss = loss_fn(output, batch_labels)
            
            loss = main_loss + aux_loss
            total_loss += loss.item()
            total_main_loss += main_loss.item()
            total_aux_loss += aux_loss.item()

            if args.task == 'classification':
                preds.extend(torch.sigmoid(output).cpu().numpy().tolist())
            else:
                preds.extend(output.cpu().numpy().tolist())
            
            labels.extend(batch_labels.cpu().numpy().tolist())
            com_ids.extend(batch['com_id'])
            pro_ids.extend(batch['pro_id'])

    result_df = pd.DataFrame({
        'com_id': com_ids,
        'pro_id': pro_ids,
        'pred': preds,
        args.label: labels
    })

    metrics = calculate_performance(result_df, args)
    
    if args.print_out:
        print(f"[{split}] Total Loss: {total_loss:.4f} | Main Loss: {total_main_loss:.4f} | Aux Loss: {total_aux_loss:.4f}")

    if wandb is not None and wandb.run is not None:
        wandb.log({
            f'{split} Total Loss': total_loss,
            f'{split} Main Loss': total_main_loss,
            f'{split} Aux Loss': total_aux_loss
        }, step=epoch)
        wandb.log({f'{split} {args.metrics[i]}': metrics[i] for i in range(len(metrics))}, step=epoch)

    return result_df, pd.DataFrame([metrics], columns=args.metrics)
