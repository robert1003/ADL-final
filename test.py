import os
import sys
import argparse
import glob
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import Model
from _QA import QAExamples_to_QAFeatureDataset
from preprocess import preprocess_parent
from postprocess2 import post_process_blend
from output import Output
from joblib import Parallel, delayed, parallel_backend

def Arg():
    args = {
        'train': 'conv',
        'cuda': 0,
        'pretrained_model': 'bert-base-multilingual-cased',
        'batch_size': 32,
        'output_file': 'prediction.csv',
        'kernel_size': [7, 7, 7, 7],
        'merge_type': 0,
        'null_threshold': [0.4, 0.4, 0.6, 0.1]
    }
    args = argparse.Namespace(**args)
    return args

def get_logits(args, dataloader, kernel_size, checkpoint, device):
    model = Model(args.pretrained_model, model_type=args.train, kernel_size=kernel_size)
    model.load_state_dict(torch.load(checkpoint, map_location=device)['state_dict'])
    
    print(f'{checkpoint} predicting...')
    raw_start_logits = []
    raw_end_logits = []
    step = 0
    model = model.to(device)
    with torch.no_grad():
        model.eval()
        for (input_ids, attention_mask, token_type_ids) in dataloader:
            step += 1
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)

            output = model(input_ids, attention_mask, token_type_ids)
            raw_start_logits.append(output[0].cpu())
            raw_end_logits.append(output[1].cpu())

            print('Iter {}/{}'.format(step, len(dataloader)), end='\r')
    raw_start_logits = torch.cat(raw_start_logits, dim=0)
    raw_end_logits = torch.cat(raw_end_logits, dim=0)
    print('...Done')

    del model
    with torch.cuda.device(device):
        torch.cuda.empty_cache()

    return raw_start_logits, raw_end_logits

def main():
    args = Arg()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:%d" % args.cuda if use_cuda else "cpu")
    BATCH_SIZE = args.batch_size
    data = sys.argv[1]
    checkpoints = [
            'loss_parent_sampler_higher_ratio.pth', 
            'loss_parent_sampler_longer.pth', 
            'loss_parent_sampler_meow.pth', 
            'loss_parent_sampler_ratio_8.0.pth'
        ]
    
    # set up tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    
    # load data
    data_path = os.path.join(data, 'ca_data/*')
    
    print('Load data...', file=sys.stderr)
    QAExamples, lines = preprocess_parent(data_path, tokenizer, 0, 1, merge_type=args.merge_type)
    QAFeatures, QAdataset = QAExamples_to_QAFeatureDataset(QAExamples, tokenizer, 'test')
    dataloader = DataLoader(QAdataset, batch_size=BATCH_SIZE, shuffle=False)
    print('DONE', file=sys.stderr)
    
    # predict
    all_raw_start_logits = []
    all_raw_end_logits = []
    with parallel_backend('loky', n_jobs=2):
        all_logits = Parallel()(delayed(get_logits)(args, dataloader, kernel_size, checkpoint, device) for checkpoint, kernel_size in zip(checkpoints, args.kernel_size))
    for raw_start_logits, raw_end_logits in all_logits:
        all_raw_start_logits.append(raw_start_logits)
        all_raw_end_logits.append(raw_end_logits)

    print('Postprocessing...')
    prediction = post_process_blend(all_raw_start_logits, all_raw_end_logits, QAExamples, QAFeatures, tokenizer, null_thresholds=args.null_threshold)
    print('Done')

    # output file
    Output(prediction, lines, args.output_file)


if __name__ == '__main__':
    main()
