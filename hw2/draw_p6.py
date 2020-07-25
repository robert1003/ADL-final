# setup environment
import os
#os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# import packages
import json
import argparse
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertForQuestionAnswering
from _utils import read_dev_examples, post_process, write_prediction
from _QA import QAExamples_to_QAFeatureDataset

# setup args
args = {
    'dev_fpath': sys.argv[1],
    'pic_fpath': sys.argv[2],
    'python_path': sys.argv[3],
    'scorer_fpath': sys.argv[4],
    'out_fpath': sys.argv[5],
    'result_fpath': sys.argv[6],
    'ckipdata_dir': sys.argv[7],
    'model_fpath': 'final/model',
    'threshold_list': [0.1, 0.3, 0.5, 0.7, 0.9],
    'batch_size': 128,
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
dev_examples = read_dev_examples(args.dev_fpath)
dev_features, dev_set = QAExamples_to_QAFeatureDataset(dev_examples, tokenizer, 'dev')
print(' ... done')
print('')

# do prediction
results = {}
for threshold in args.threshold_list:
    print(threshold)
    result = post_process(
        model=model, 
        examples=dev_examples, 
        features=dev_features, 
        dataset=dev_set, 
        batch_size=args.batch_size, 
        tokenizer=tokenizer, 
        threshold=threshold, 
        max_len=args.max_len,
        device=args.device
    )
    write_prediction(result, args.out_fpath)
    os.system('{} {} {} {} {} {}'.format(
        args.python_path,
        args.scorer_fpath,
        args.dev_fpath,
        args.out_fpath,
        args.result_fpath,
        args.ckipdata_dir
    ))
    with open(args.result_fpath, 'r') as f:
        score = json.loads(f.readline().strip())
    results[threshold] = score

# get result
f1 = {
    'overall': [],
    'answerable': [],
    'unanswerable': []
}
em = {
    'overall': [],
    'answerable': [],
    'unanswerable': []
}
for threshold in args.threshold_list:
    f1['overall'].append(results[threshold]['overall']['f1'])
    f1['answerable'].append(results[threshold]['answerable']['f1'])
    f1['unanswerable'].append(results[threshold]['unanswerable']['f1'])

    em['overall'].append(results[threshold]['overall']['em'])
    em['answerable'].append(results[threshold]['answerable']['em'])
    em['unanswerable'].append(results[threshold]['unanswerable']['em'])

# draw pic
fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10,5))

# plot f1
axes[0].set_title('F1')
axes[0].set_xticks(args.threshold_list)
l1 = axes[0].plot(args.threshold_list, f1['overall'], marker='o')
l2 = axes[0].plot(args.threshold_list, f1['answerable'], marker='o')
l3 = axes[0].plot(args.threshold_list, f1['unanswerable'], marker='o')

# plot f2
axes[1].set_title('EM')
axes[1].set_xticks(args.threshold_list)
axes[1].plot(args.threshold_list, em['overall'], marker='o')
axes[1].plot(args.threshold_list, em['answerable'], marker='o')
axes[1].plot(args.threshold_list, em['unanswerable'], marker='o')

# draw legend
fig.legend(
    [l1, l2, l3],     
    labels=['overall', 'answerable', 'unanswerable'],   
    loc='upper right',
)

# set title and x, y label
plt.suptitle('Performance on Different Threshold')
fig.text(0.5, 0.04, 'Answerable Threshold', ha='center')
fig.text(0.04, 0.5, 'Score', va='center', rotation='vertical')

# save it
plt.savefig(args.pic_fpath)
