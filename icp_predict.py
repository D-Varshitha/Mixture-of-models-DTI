import os
import torch
import pandas as pd
import numpy as np
import itertools
import argparse

from sklearn.metrics import roc_auc_score as auc
from sklearn.metrics import precision_recall_curve as pre_rec_c
from sklearn.metrics import auc as auc_c
from sklearn.metrics import accuracy_score as acc

def argument_parse():
  p = argparse.ArgumentParser()
  p.add_argument('--d', type=str, default='kinome')
  return p.parse_args()

def calculate_nonconformity(pred, lab):
  return abs(pred - lab)

def calculate_pvalue(cal_nonConfs, test_nonConf):  
  n = 0
  for cal_nonConf in cal_nonConfs + [test_nonConf]:
    if cal_nonConf >= test_nonConf:
      n += 1
  return n/(len(cal_nonConfs)+1)

def generate_pred_for_test(test_pred, cal_nonConfs):
  p_0 = calculate_pvalue(cal_nonConfs, calculate_nonconformity(test_pred, 0))
  p_1 = calculate_pvalue(cal_nonConfs, calculate_nonconformity(test_pred, 1))
  test_pre_label = np.argmax([p_0, p_1])
  return test_pre_label, [p_0, p_1][test_pre_label]


if __name__ == '__main__':
  args = argument_parse()
  d = args.d

  df = pd.read_csv(f'results/new_ensdti_internal_validation_on_{d}.csv',index_col=[0,1])
  models = df.columns.to_list()
  models.remove('label')

  cal_df_frac = 0.5
  cal_df = df.sample(frac=cal_df_frac)
  test_df = pd.concat([df, cal_df]).drop_duplicates(keep=False)

  for model in models:
    calNonConfs = []
    testPredictions = []
    Confs = []
    for i,r in cal_df.iterrows():
      calNonConfs.append(calculate_nonconformity(r[f'{model}'], r['label']))
    for i,r in test_df.iterrows():
      tp, conf = generate_pred_for_test(r[f'{model}'], calNonConfs)
      testPredictions.append(tp)
      Confs.append(conf)
    
    test_df[f'{model}_conf'] = Confs
    test_df[f'{model}_predByICP'] = testPredictions
  result_file = f'results/ICP_on_{d}.csv'
  print(f'Thre\tModel\tCount\tError Rates')
  print(f'Threshold\tModel\tCount\tError Rates\tCorrect numbers\tAccuracy', file=open(result_file, 'w'))
  for threshold in [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]:
    for model in models:
      sub_testdf = test_df[test_df[f'{model}_conf']>threshold] 
      count = len(sub_testdf)
      error_rates = 0
      for i,r in sub_testdf.iterrows():
        if r[f'{model}_predByICP'] != r['label']:
          error_rates += 1
      error_rates /= len(sub_testdf)
      correct_rates = 1 - error_rates
      correct_count = int(len(sub_testdf)*correct_rates)
      print('%.2f\t%s\t%i\t%.3f\t%i\t%.3f'%(threshold, model, count, error_rates, correct_count, correct_rates))
      print('%.2f\t%s\t%i\t%.3f\t%i\t%.3f'%(threshold, model, count, error_rates, correct_count, correct_rates), file=open(result_file, 'a'))
    print('------------------------------------')