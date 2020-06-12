# setup environment
import os
#os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# import packages
import json
import sys
import argparse
import random
import torch
import numpy as np
from transformers import BertTokenizer, BertForQuestionAnswering
from _utils import read_train_examples, train
from _QA import QAExamples_to_QAFeatureDataset

# set random seed
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True

# setup args
args = {
    'train_fpath': sys.argv[1],
    'model_fpath': 'final/bert.pth',
    'device': 'cuda'
}
args = argparse.Namespace(**args)

# load model & tokenizer
print('[*] Loading model & tokenizer ... ', end='')
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertForQuestionAnswering.from_pretrained('bert-base-multilingual-cased').to(args.device)
print('done')

# load and process data
print('[*] Load and process data ... ')
train_examples = read_train_examples(args.train_fpath)
train_features, train_set = QAExamples_to_QAFeatureDataset(train_examples, tokenizer, 'train')
print(' ... done')
print('')

# train!
# part 1
print('[*] Start part 1 training ... ')
train(
    model, 
    dataset=train_set, 
    batch_size=4, 
    learning_rate=3e-5,
    adam_epsilon=1e-8,
    epochs=10,
    gradient_accumulation_steps=8, 
    max_grad_norm=1.0,
    model_fname=args.model_fpath,
    helper_fname=os.path.join(args.model_fpath, 'helper.pth'),
    device=args.device
)
print('[*] ... done')
print('')

