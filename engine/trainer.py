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

def train(train_loader, model, loss_fn, optimizer, args, valid_loader=None):
    best_perf = None
    for epoch in trange(args.epoch, desc="Training"):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            output, labels = forward_batch(model, batch, args.model, args.device)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if wandb is not None and wandb.run is not None:
            wandb.log({f'Train Loss': epoch_loss}, step=epoch)

        if args.lr_decay and args.decay_interval and epoch % args.decay_interval == 0:
            optimizer.param_groups[0]['lr'] *= args.lr_decay

        if valid_loader:
            result_df, perf_df = test(valid_loader, model, loss_fn, args, split='Valid', epoch=epoch)
            if args.save_model:
                save_model(model, args, epoch, result_df=result_df)

    return model



def test(loader, model, loss_fn, args, split='Test', epoch=0):
    model.eval()
    preds, labels, com_ids, pro_ids = [], [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            output, batch_labels = forward_batch(model, batch, args.model, args.device)
            loss = loss_fn(output, batch_labels)
            total_loss += loss.item()

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
    if wandb is not None and wandb.run is not None:
        wandb.log({f'{split} Loss': total_loss}, step=epoch)
        wandb.log({f'{split} {args.metrics[i]}': metrics[i] for i in range(len(metrics))}, step=epoch)

    return result_df, pd.DataFrame([metrics], columns=args.metrics)

def forward_batch(model, batch, model_name, device):
    if model_name == 'dp':
        com_af = batch['com_af'].to(device)
        com_bf = batch['com_bf'].to(device)
        com_ag = batch['com_ag'].to(device)
        com_bg = batch['com_bg'].to(device)
        com_abn = batch['com_abn'].to(device)
        pro = batch['pro_feat'].to(device)
        labels = batch['label'].to(device)
        output = model([com_af, com_bf, com_ag, com_bg, com_abn], pro).reshape(-1)
    elif model_name == 'cpi':
        com = batch['com_feat'].to(device)
        adj = batch['com_adj'].to(device)
        pro = batch['pro_feat'].to(device)
        labels = batch['label'].to(device)
        output = model(com, adj, pro).reshape(-1)
    else:
        com = batch['com_feat'].to(device)
        pro = batch['pro_feat'].to(device)
        labels = batch['label'].to(device)
        output = model(com, pro).reshape(-1)
    return output, labels
