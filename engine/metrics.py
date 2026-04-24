import pandas as pd
import numpy as np

from math import sqrt
from scipy import stats

from sklearn.metrics import roc_auc_score as auc
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import recall_score as rec
from sklearn.metrics import precision_score as pre
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import balanced_accuracy_score as ba
from sklearn.metrics import average_precision_score as auprc

def calculate_performance(df, args):
    if args.task == 'classification':
        return calculate_performance_classification(df,args)
    else:
        return calculate_performance_regression(df,args)


# 'Acc','Pre','Rec','Spe','AUC','BA'
def calculate_performance_classification(df,args): # df with columns 'label', 'pred'
    # Raw probabilities for AUC
    prob = np.array(df['pred'])
    # Rounded predictions for Acc, Pre, Rec, etc.
    pred = prob.round().tolist()
    label = df[args.label]
    
    acc_ = acc(label, pred)
    pre_ = pre(label, pred)
    rec_ = rec(label, pred)
    try:
        auc_ = auc(label, prob)
    except Exception:
        auc_ = 0.0
    try:
        auprc_ = auprc(label, prob)
    except Exception:
        auprc_ = 0.0
    ba_ = ba(label, pred)
    tn, fp, fn, tp = cm(label, pred, labels=[0, 1]).ravel()
    spe_ = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return [acc_, pre_, rec_, spe_, auc_, auprc_, ba_]

def calculate_icp_metrics_classification(df, args, threshold):
    """
    Calculates Validity (Coverage) and Efficiency (Set Size) for ICP.
    """
    prob = np.array(df['pred'])
    label = np.array(df[args.label])
    
    # Non-conformity score for class 1: 1 - p
    # Non-conformity score for class 0: p
    
    # Class 0 is in set if prob <= threshold
    # Class 1 is in set if (1 - prob) <= threshold  => prob >= 1 - threshold
    
    in_set_0 = prob <= threshold
    in_set_1 = prob >= (1.0 - threshold)
    
    set_size = in_set_0.astype(int) + in_set_1.astype(int)
    
    # Coverage: true label is in the set
    coverage = np.where(label == 1, in_set_1, in_set_0)
    
    avg_coverage = np.mean(coverage)
    avg_set_size = np.mean(set_size)
    
    return [avg_coverage, avg_set_size]


def calculate_performance_regression(df, args):
    pred  = np.array(df['pred'])
    label = np.array(df[args.label])
    mse_      = mse(label, pred)
    rmse_     = rmse(label, pred)
    # pearson/spearman return NaN when predictions are constant (e.g. early epochs)
    try:
        pearson_  = pearson(label, pred)
        if not np.isfinite(pearson_): pearson_ = 0.0
    except Exception:
        pearson_ = 0.0
    try:
        spearman_ = spearman(label, pred)
        if not np.isfinite(spearman_): spearman_ = 0.0
    except Exception:
        spearman_ = 0.0
    ci_       = ci(label, pred)
    return [mse_, rmse_, pearson_, spearman_, ci_]

def calculate_icp_metrics_regression(df, args, threshold):
    """
    Calculates Coverage and Average Interval Width for ICP.
    """
    pred = np.array(df['pred'])
    label = np.array(df[args.label])
    
    # Interval is [pred - threshold, pred + threshold]
    lower = pred - threshold
    upper = pred + threshold
    
    coverage = (label >= lower) & (label <= upper)
    avg_coverage = np.mean(coverage)
    avg_width = 2 * threshold
    
    return [avg_coverage, avg_width]

def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse
def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse
def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp
def spearman(y,f):
    rs = stats.spearmanr(y, f)[0]
    return rs
def ci(y, f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y) - 1
    j = i - 1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z + 1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i - 1
    return S / z if z > 0 else 0.0   # guard: z=0 when all labels are identical