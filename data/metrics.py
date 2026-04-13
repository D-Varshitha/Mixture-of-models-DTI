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

def calculate_performance(df, args):
    if args.task == 'classification':
        return calculate_performance_classification(df,args)
    else:
        return calculate_performance_regression(df,args)


# 'Acc','Pre','Rec','Spe','AUC','BA'
def calculate_performance_classification(df,args): # df with columns 'label', 'pred'
    # print(df)
    pred = np.array(df['pred']).round().tolist()
    label = df[args.label]
    acc_ = acc(label, pred)
    pre_ = pre(label, pred)
    rec_ = rec(label, pred)
    auc_ = auc(label, pred)
    ba_ = ba(label, pred)
    tn, fp, fn, tp = cm(label, pred).ravel()
    spe_ = tn / (tn + fp)
    
    return [acc_, pre_, rec_, spe_, auc_, ba_]


# 'MSE', 'RMSE', 'pearson', 'spearman', 'CI',
def calculate_performance_regression(df,args): # df with columns 'label', 'pred'
    pred = np.array(df['pred'])
    label = np.array(df[args.label])
    rmse_ = rmse(label, pred)
    mse_ = mse(label, pred)
    pearson_ = pearson(label, pred)
    spearman_ = spearman(label, pred)
    ci_ = ci(label, pred)
    return mse_, rmse_, pearson_, spearman_, ci_

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
def ci(y,f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci