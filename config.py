import argparse

MODELS = ['dcdti', 'dpdta', 'mdprd', 'dp', 'cpi', 'mlp', 'rf', 'svm','perceivercpi', 'smtdta', 'ensdti']
metrics_classification = ['Acc','Pre','Rec','Spe','AUC','BA']
metrics_regression = ['MSE', 'RMSE', 'pearson', 'spearman', 'CI']


parser = argparse.ArgumentParser(description='Specify the experiments settings of EnsDTI')

parser.add_argument('--data', metavar='-d', default=None, type=str)
parser.add_argument('--train-data', default=None, type=str)
parser.add_argument('--test-data', default=None, type=str)
parser.add_argument('--split-method', metavar='-s', default='random', choices=['random', 'none'], type=str)
# parser.add_argument('--shuffle', default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--model', metavar='-m', default='default', type=str)
parser.add_argument('--exp-mode', default='5_fold_val', choices=['pure_train','pure_test','train_and_test','5_fold_val', 'train_wo_valid'], type=str)
parser.add_argument('--model-dir', metavar='-d', default='davis/model/', type=str)
parser.add_argument('--save-perf', metavar='-p', default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--save-result', metavar='-r', default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--save-model', metavar='-r', default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--epoch', metavar='-e', default=100, type=int)
parser.add_argument('--batch', metavar='-b', default=128, type=int)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--device',default='0',type=str)
parser.add_argument('--split-type', default='random', type=str)
parser.add_argument('--get-dataset', default='generate',choices=['load','generate'])
parser.add_argument('--print-out', default=True, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument('--lr-decay', default=None, type=float)
parser.add_argument('--weight-decay', default=0, type=float)
parser.add_argument('--decay-interval', default=None, type=int)
parser.add_argument('--label', default='label', type=str)
parser.add_argument('--task', default='classification', type=str, choices=['classification', 'regression'])
parser.add_argument('--custom', default='', type=str, help='customized title for tsne plot')
parser.add_argument('--com-len', default=100, type=int, help='customized com len')
parser.add_argument('--pro-len', default=1000, type=int, help='customized pro len')
parser.add_argument('--model-mode', default='train_wo_valid', type=str)

args = parser.parse_args()
print('ARGUMENT:\n', args) 
