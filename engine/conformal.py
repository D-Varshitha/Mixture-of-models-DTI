import numpy as np
import torch


# -------------------------------
# 1. Calibration Scores
# -------------------------------
def get_calibration_scores(model, loader, task):
    model.eval()
    scores = []

    with torch.no_grad():
        for batch in loader:
            output, _ = model(batch)

            labels = batch['label'].to(output.device).float().view(-1)
            preds = output.view(-1)

            if task == 'classification':
                probs = torch.sigmoid(preds)
                batch_scores = torch.abs(probs - labels)
            else:
                batch_scores = torch.abs(preds - labels)

            scores.extend(batch_scores.cpu().numpy())

    return np.array(scores)


# -------------------------------
# 2. P-value (Classification ICP)
# -------------------------------
def calculate_p_value(cal_scores, test_score):
    n = len(cal_scores)
    count = np.sum(cal_scores >= test_score)
    return (count + 1) / (n + 1)


# -------------------------------
# 3. ICP Regression (FIXED)
# -------------------------------
def apply_icp_regression(output, cal_scores, alpha=0.1, q=None):
    preds = output.view(-1).cpu().numpy()

    # Compute quantile once
    if q is None:
        q = np.quantile(cal_scores, 1 - alpha)

    lower = preds - q
    upper = preds + q

    # Return structured output (better than list of tuples)
    return {
        "pred": preds,
        "lower": lower,
        "upper": upper,
        "q": q
    }


# -------------------------------
# 4. Unified ICP Logic
# -------------------------------
def apply_icp_reference_logic(output, cal_scores, task, alpha=0.1, q=None):

    if task == 'classification':
        probs = torch.sigmoid(output).view(-1).cpu().numpy()

        results = []
        for p in probs:
            p_0 = calculate_p_value(cal_scores, p)
            p_1 = calculate_p_value(cal_scores, 1.0 - p)

            pred_label = 1 if p_1 > p_0 else 0
            confidence = max(p_0, p_1)

            results.append((pred_label, confidence))

        return results

    else:
        return apply_icp_regression(output, cal_scores, alpha, q)