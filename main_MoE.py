import torch
import os
import sys
import wandb
from torch import nn
from tqdm import trange

from data import split_dataset_by_fold
from data.moe_dataset import MoEDataset
from models import build_model
from models.model_MoE import DTI_Sparse_MoE
from engine.trainer_MoE import train_moe, test_moe
from config import args, metrics_classification, metrics_regression


def main():

    # Automatic Hardware Detection (Removes strict exit from old code)
    if torch.cuda.is_available() and args.device != 'cpu':
        device = torch.device('cuda')
        print("Hardware detected: GPU")
    else:
        device = torch.device('cpu')
        print("Hardware detected: CPU. This is great for local testing!")
        
    torch.manual_seed(42)

    data_name = args.data if args.data else 'davis'

    # 1. Initialize Dataset
    print(f"Loading Dataset: {data_name}...")
    # NOTE: In debug mode we do NOT build monolithic precomputed features if they don't exist
    label_col = 'lab' if args.task == 'classification' else 'affinity'
    dataset = MoEDataset(root=os.getcwd(), dataset_name=data_name, label_type=label_col, mode='load' if args.mode == 'full' else 'debug')
    
    # 2. Handle Debug Mode vs Full Mode
    if args.mode == 'debug':
        print(f"DEBUG MODE: Selecting subset of {args.subset_size} samples.")
        subset_indices = torch.randperm(len(dataset))[:args.subset_size].tolist()
        dataset = torch.utils.data.Subset(dataset, subset_indices)
        all_idx = list(range(len(dataset)))
        
        # Override epoch & batch for debug to prevent silent CPU Out-Of-Memory crash
        args.epoch = min(args.epoch, 2)
        args.batch = min(args.batch, 4)
    else:
        all_idx = list(range(len(dataset)))
        
    if args.task == 'classification':
        loss_fn = nn.BCEWithLogitsLoss()
        args.metrics = metrics_classification
    else:
        loss_fn = nn.MSELoss()
        args.metrics = metrics_regression
        
    # 3. Build MoE
    print("Building MoE Model...")
    
    # Pre-instantiate the 6 exactly required experts
    # Data is passed None here for simplicity; if experts strictly mandate dynamic input sizing, 
    # we must parse dataset to establish `fp_num` & `word_num` appropriately.
    # We will pass a dummy `data` object to builder for CPI initialization
    class DummyData:
        def __init__(self, cpi_fp_dict=10648, cpi_words=10648):
            self.fp_num = cpi_fp_dict
            self.word_num = cpi_words
    
    dummy_data = DummyData()
    
    experts_dict = {
        'dpdta': build_model('dpdta', args.task),
        'dcdti': build_model('dcdti', args.task),
        'dp': build_model('dp', args.task),
        'mdprd': build_model('mdprd', args.task),
        'gifdti': build_model('gifdti', args.task),
        'perceivercpi': build_model('perceivercpi', args.task)
    }
    
    moe_model = DTI_Sparse_MoE(experts_dict, drug_vocab=65, prot_vocab=26, k=2, lambda_aux=args.lambda_aux)
    moe_model = moe_model.to(device)
    
    optimizer = torch.optim.Adam(moe_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 4. K-Fold Execution Pipeline
    print(f"Execution Pipeline starting. Mode={args.mode}")
    # Always do fold 0 for debug, 5-folds for full mode if exp_mode is 5_fold_val
    folds_to_run = 5 if args.mode == 'full' and args.exp_mode == '5_fold_val' else 2
    
    for fold in range(folds_to_run):
        print(f"\n===== FOLD {fold} =====")
        
        # We reuse existing data split functionality cleanly
        from data.data_split import split_dataset_by_fold
        train_loader, valid_loader, test_loader = split_dataset_by_fold(dataset, all_idx, fold, args.batch)
        
        if wandb.run is not None:
             wandb.init(project='ensdti-moe', name=f'MoE_{data_name}_fold{fold}', config=args, reinit=True)
             
        # Train MoE Pipeline
        moe_model = train_moe(train_loader, moe_model, loss_fn, optimizer, args, valid_loader=valid_loader)
        
        # Test MoE Pipeline
        print(f"\nFinal Testing for FOLD {fold}")
        result, perf = test_moe(test_loader, moe_model, loss_fn, args, split='Test_Final')
        
        if args.save_result:
            out_path = f'{os.getcwd()}/dataset/{data_name}/moe/{args.exp_mode}/{args.custom}/fold_{fold}'
            os.makedirs(out_path, exist_ok=True)
            result.to_csv(f'{out_path}/test_result.csv', index=False)
            if args.save_perf:
                perf.to_csv(f'{out_path}/test_perf.csv', index=False)

if __name__ == '__main__':
    main()
