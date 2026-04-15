import torch
import os
import sys
try:
    import wandb
except ImportError:
    wandb = None
from torch import nn
from tqdm import trange

from data import prepare_data, return_dataloader, split_dataset_by_fold
from models import declare_model
from engine.trainer import train, test
from config import args, MODELS, metrics_classification, metrics_regression


def main():

    if args.device != 'cpu':
        device = torch.device('cuda')
    else:
        if args.exp_mode != 'pure_test':
            sys.exit('Do not support training model with cpu')
        device = torch.device('cpu')
    torch.manual_seed(42)

    # Load data path based on mode
    if args.exp_mode == '5_fold_val':
        data = args.data
    elif args.exp_mode in ['pure_train', 'train_wo_valid']:
        data = args.train_data
    elif args.exp_mode == 'pure_test':
        data = args.test_data
    elif args.exp_mode == 'train_and_test':
        data = [args.train_data, args.test_data]
    else:
        sys.exit('Invalid exp_mode specified')

    # Define model list based on args.model
    if args.model == 'ensdti':
        models = 'ensdti'
    elif args.model == 'default':
        models = MODELS
    elif args.model.startswith('ablation'):
        modelsToRemove = args.model.replace("ablation_", "").split("_")
        models = [m for m in MODELS if m not in modelsToRemove]
    elif '_' in args.model:
        models = args.model.split('_')
    else:
        models = [args.model]

    # Prepare data
    data_obj = prepare_data(data, models, args)

    # Training or testing logic
    for m in models:
        if args.task == 'classification':
            # FIX 1 (repo-wide): prefer logits + BCEWithLogitsLoss to avoid double-sigmoid.
            # Expert models are now expected to return raw logits.
            loss_fn = nn.BCEWithLogitsLoss()
            args.metrics = metrics_classification
        else:
            loss_fn = nn.MSELoss()
            args.metrics = metrics_regression

        if args.exp_mode == 'pure_test':
            test_loader = return_dataloader(data_obj[m], args.batch, shuffle=False)
            model.load_state_dict(torch.load(f'{os.getcwd()}/dataset/{args.data}/{m}/{args.model_mode}/{args.custom}/final_model_{args.custom}.pt'))
            result, perf = test('Test', test_loader, model, loss_fn, args)
        else:
            if args.exp_mode == '5_fold_val':
                datasets, all_idx = data_obj
                for fold in range(5):
                    # FIX 5: Re-initialize model + optimizer inside fold loop.
                    model = declare_model(m, args.task, data_obj if m in ['dpdta', 'cpi'] else None).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                    train_loader, valid_loader, test_loader = split_dataset_by_fold(datasets[m], all_idx, fold, args.batch)
                    if wandb is not None:
                        wandb.init(project='ensdti', name=f'{m}_fold{fold}', config=args, reinit=True)
                    train(train_loader, model, loss_fn, optimizer, args, valid_loader=valid_loader)
                    result, perf = test('Test', test_loader, model, loss_fn, args)
            elif args.exp_mode in ['pure_train', 'train_and_test']:
                model = declare_model(m, args.task, data_obj if m in ['dpdta', 'cpi'] else None).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                train_datasets, valid_datasets = data_obj[:2]
                train_loader = return_dataloader(train_datasets[m], args.batch, shuffle=True)
                valid_loader = return_dataloader(valid_datasets[m], args.batch, shuffle=False)
                if wandb is not None:
                    wandb.init(project='ensdti', name=f'{m}_train', config=args, reinit=True)
                train(train_loader, model, loss_fn, optimizer, args, valid_loader=valid_loader)
                if args.exp_mode == 'train_and_test':
                    test_loader = return_dataloader(data_obj[2][m], args.batch, shuffle=False)
                    result, perf = test('Test', test_loader, model, loss_fn, args)

        # Save results if needed
        if args.save_result:
            out_path = f'{os.getcwd()}/dataset/{args.data}/{m}/{args.exp_mode}/{args.custom}'
            os.makedirs(out_path, exist_ok=True)
            result.to_csv(f'{out_path}/test_result.csv', index=False)
        if args.save_perf:
            perf.to_csv(f'{out_path}/test_perf.csv', index=False)

if __name__ == '__main__':
    main()