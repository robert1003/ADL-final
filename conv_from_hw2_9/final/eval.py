# setup environment
#import os
#os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# import packages
import json
import argparse
import sys
from transformers import BertTokenizer, BertForQuestionAnswering
from _utils import read_test_examples, post_process, write_prediction
from _QA import QAExamples_to_QAFeatureDataset

# setup args
args = {
    'test_fpath': sys.argv[1],
    'output_fpath': sys.argv[2],
    'model_fpath': 'final/model',
    'batch_size': 128,
    'threshold': 0.5,
    'max_len': 30,
    'device': 'cuda'
}
args = argparse.Namespace(**args)

# load model & tokenizer
print('[*] Loading model & tokenizer ... ', end='')
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForQuestionAnswering.from_pretrained(args.model_fpath).to(args.device)
print('done')

# load and process data
print('[*] Load and process data ... ')
test_examples = read_test_examples(args.test_fpath)
test_features, test_set = QAExamples_to_QAFeatureDataset(test_examples, tokenizer, 'test')
print(' ... done')
print('')

# predict!
results = post_process(
    model=model, 
    examples=test_examples, 
    features=test_features, 
    dataset=test_set, 
    batch_size=args.batch_size, 
    tokenizer=tokenizer, 
    threshold=args.threshold, 
    max_len=args.max_len,
    device=args.device
)

# write prediction
write_prediction(results, args.output_fpath)
