import numpy as np
import torch

def get_conformal_threshold(model, loader, alpha, task):
    """
    Calculates the non-conformity score threshold for Inductive Conformal Prediction.
    
    For classification: Non-conformity score = 1 - P(correct_class)
    For regression: Non-conformity score = |y - y_pred|
    """
    model.eval()
    scores = []
    
    with torch.no_grad():
        for batch in loader:
            output, _ = model(batch)
            labels = batch['label'].to(output.device).float()
            
            if task == 'classification':
                # P(y=1)
                probs = torch.sigmoid(output)
                # Non-conformity score for the TRUE label:
                # If label=1, score is 1 - P(y=1)
                # If label=0, score is 1 - P(y=0) = 1 - (1 - P(y=1)) = P(y=1)
                batch_scores = torch.where(labels == 1, 1.0 - probs, probs)
                scores.extend(batch_scores.cpu().numpy().tolist())
            else:
                # Absolute error for regression
                batch_scores = torch.abs(output - labels)
                scores.extend(batch_scores.cpu().numpy().tolist())
    
    scores = np.array(scores)
    # The threshold is the (1-alpha) quantile of the non-conformity scores
    # We use q = (n+1)(1-alpha) / n  approximation or similar.
    # Standard ICP: find the smallest q such that at least (1-alpha) fraction of scores are <= q.
    n = len(scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(max(q_level, 0.0), 1.0)
    
    threshold = np.quantile(scores, q_level, method='higher')
    return threshold

def apply_conformal_prediction(output, threshold, task):
    """
    Applies the threshold to produce prediction sets or intervals.
    """
    if task == 'classification':
        probs = torch.sigmoid(output).cpu().numpy()
        # For each sample, the prediction set includes class '1' if 1-P(y=1) <= threshold
        # and class '0' if P(y=1) <= threshold.
        # This is equivalent to:
        # Include class 1 if P(y=1) >= 1 - threshold
        # Include class 0 if P(y=1) <= threshold
        
        pred_sets = []
        for p in probs:
            s = []
            if p <= threshold: s.append(0)
            if p >= (1.0 - threshold): s.append(1)
            pred_sets.append(s)
        return pred_sets
    else:
        preds = output.cpu().numpy()
        # Interval is [pred - threshold, pred + threshold]
        return [[p - threshold, p + threshold] for p in preds]
