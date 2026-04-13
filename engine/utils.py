import os
import torch

def save_model(model, args, epoch, result_df=None):
    exp_dir = os.path.join(os.getcwd(), 'dataset', args.data, args.model, args.exp_mode, str(args.custom))
    os.makedirs(exp_dir, exist_ok=True)
    model_path = os.path.join(exp_dir, f'epoch{epoch}_model.pt')
    torch.save(model.state_dict(), model_path)

    if result_df is not None:
        result_df.to_csv(os.path.join(exp_dir, f'epoch{epoch}_results.csv'), index=False)
