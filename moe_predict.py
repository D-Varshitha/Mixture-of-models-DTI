import os
import joblib
import argparse
import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
folds = 5
models = ['dcdti','dpdta','mdprd','dp','perceivercpi','smtdta']
param = {
  'max_iter':[15, 20, 25, 30],
  'hidden_layer_sizes':[10, 15, 20],
  'solver':['adam'],
  'learning_rate_init':[0.01, 0.001, 0.0001, 0.00001]
}

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", '-d', help="Customized data dir name", default='test_data')
    dataset = parser.parse_args().dataset

    input_csv = []
    for model in models:
        re_csv = pd.read_csv(f'tmp/{dataset}/{model}_result.csv', index_col=[0,1])
        input_csv.append(re_csv)
    input_csv = pd.concat(input_csv, axis=1)
    input_csv.to_csv(f'tmp/{dataset}/all_prob.csv')

    train_index = pd.read_csv(f'tmp/{dataset}/train_index.csv', header=None).values.flatten()
    test_index = pd.read_csv(f'tmp/{dataset}/test_index.csv', header=None).values.flatten()

    result = []
    labels = []
    train_X = input_csv.iloc[train_index,:-1]
    train_y = input_csv.iloc[train_index,-1]
    test_X = input_csv.iloc[test_index,:-1]
    test_y = input_csv.iloc[test_index,-1]
    model = MLPClassifier(early_stopping=True, activation='logistic')
    clf = GridSearchCV(model, param)
    clf.fit(train_X, train_y)

    tra_pre = clf.predict(train_X)
    tes_pre = clf.predict(test_X)

    result.append(clf.predict_proba(test_X))
    labels += test_y.tolist()

    result = model.predict_proba(input_csv)
    result_csv = pd.DataFrame(result, index=input_csv.index)
    result_csv['label'] = labels
    
    file_to_save = f'output/{dataset}/result.csv'
    if not os.path.exists(os.path.dirname(file_to_save)):
        os.makedirs(os.path.dirname(file_to_save))
    result_csv.to_csv(file_to_save ,index=True)
    print('Successfully generated final prediction!')