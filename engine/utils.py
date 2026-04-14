import os
import torch

def save_model(model, args, epoch, result_df=None):
    safe_data = str(args.data) if args.data else "default_data"
    safe_model = str(args.model) if args.model else "default_model"
    safe_exp = str(args.exp_mode) if args.exp_mode else "default_exp"
    safe_custom = str(args.custom) if args.custom else ""
    
    exp_dir = os.path.join(os.getcwd(), 'dataset', safe_data, safe_model, safe_exp, safe_custom)
    os.makedirs(exp_dir, exist_ok=True)
    model_path = os.path.join(exp_dir, f'epoch{epoch}_model.pt')
    torch.save(model.state_dict(), model_path)

    if result_df is not None:
        result_df.to_csv(os.path.join(exp_dir, f'epoch{epoch}_results.csv'), index=False)
