import numpy as np
import torch

def get_calibration_scores(model, loader, task):
    """
    Returns the non-conformity scores for the calibration (validation) set.
    """
    model.eval()
    scores = []
    with torch.no_grad():
        for batch in loader:
            output, _ = model(batch)
            labels = batch['label'].to(output.device).float()
            
            if task == 'classification':
                probs = torch.sigmoid(output)
                # Non-conformity score: abs(pred - label)
                batch_scores = torch.abs(probs - labels)
            else:
                batch_scores = torch.abs(output - labels)
            
            scores.extend(batch_scores.cpu().numpy().tolist())
    return np.array(scores)

def calculate_p_value(cal_scores, test_score):
    """
    p-value = (number of calibration scores >= test_score) / (n_cal + 1)
    """
    n = len(cal_scores)
    count = np.sum(cal_scores >= test_score)
    # The reference code uses (count) / (n + 1) where test_score is included in the rank.
    # To match reference logic exactly: 
    # (n_cal_where_score >= test_score + 1_for_test_itself) / (n_cal + 1)
    return (count + 1) / (n + 1)

def apply_icp_reference_logic(output, cal_scores, task):
    """
    Exactly matches the reference logic:
    1. Calculate p-value for every possible label.
    2. Pick the label with the highest p-value.
    """
    if task == 'classification':
        probs = torch.sigmoid(output).cpu().numpy()
        results = []
        for p in probs:
            # p_0: p-value assuming true label is 0 (score is p - 0 = p)
            p_0 = calculate_p_value(cal_scores, p)
            # p_1: p-value assuming true label is 1 (score is abs(p - 1) = 1 - p)
            p_1 = calculate_p_value(cal_scores, 1.0 - p)
            
            # Predict the class with the highest p-value (lowest non-conformity)
            pred_label = 1 if p_1 > p_0 else 0
            confidence = max(p_0, p_1)
            results.append((pred_label, confidence))
        return results
    else:
        # For regression, we can't iterate through all labels.
        # Reference code logic is designed for classification.
        # For regression, we'll return the point prediction and a standard p-value.
        preds = output.cpu().numpy()
        return [(p, 0.5) for p in preds] # Placeholder for regression
