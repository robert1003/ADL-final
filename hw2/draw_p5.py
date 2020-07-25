# setup environment
#import os
#os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# import packages
import json
import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer
from _utils import read_train_examples
from _QA import QAExamples_to_QAFeatureDataset

# setup args
args = {
    'train_fpath': sys.argv[1],
    'pic_fpath': sys.argv[2],
    'device': 'cuda'
}
args = argparse.Namespace(**args)

# load tokenizer
print('[*] Loading tokenizer ... ', end='')
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
print('done')

# load and process data
print('[*] Load and process data ... ')
train_examples = read_train_examples(args.train_fpath)
train_features, train_set = QAExamples_to_QAFeatureDataset(train_examples, tokenizer, 'train')
print(' ... done')
print('')

# collect answer length
answer_length = []
for example, feature in zip(train_examples, train_features):
    if example.answerable:
        answer_length.append(feature.end_position - feature.start_position + 1)

# plot cumulative distribution
plt.figure(figsize=(10, 5))
kwargs = {'cumulative': True, 'rwidth':0.75}
sns.distplot(answer_length, hist_kws=kwargs, kde=False, norm_hist=True)
plt.xlabel('Length')
plt.ylabel('Count (%)')
plt.title('Cumulative Answer Length')
plt.savefig(args.pic_fpath)
