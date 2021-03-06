import os
import sys
import argparse
import torch
import torch.nn as nn
import logging
from transformers import AutoTokenizer, AutoModel, AdamW
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import tqdm
from model import Model
from _QA import QAExamples_to_QAFeatureDataset
from preprocess import *
from train import Train

def Arg():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--train', type=str, help='conv for convolution, linear for linear')
    arg_parser.add_argument('--cuda', default=0, type=int, help='cuda device number')
    arg_parser.add_argument('--model_name', type=str, required=True, help='model save name')
    arg_parser.add_argument('--log_name', type=str, required=True, help='log file name')
    arg_parser.add_argument('--train_data', type=str, default='../release/train', help='train data folder')
    arg_parser.add_argument('--dev_data', type=str, default='../release/dev', help='dev data folder')
    arg_parser.add_argument('--dev_ref_file', type=str, default='../release/dev/dev_ref.csv', help='dev ref file path')
    arg_parser.add_argument('--pretrained_model', type=str, default='bert-base-multilingual-cased', help='name of pretrained model to use')
    arg_parser.add_argument('--epochs', type=int, default=20, help='training epochs')
    arg_parser.add_argument('--learning_rate', type=float, default=3e-5, help='learning rate')
    arg_parser.add_argument('--use_sampler', action='store_true', help='use sampler to solve imbalance problem')
    arg_parser.add_argument('--ratio', type=float, default=2.0, help='use sampler to solve imbalance problem, \
            prob(answerable) / prob(unanswerable) what you enter')
    arg_parser.add_argument('--round', type=int, default=2000, help='iter of each epoch using sampler')
    arg_parser.add_argument('--kernel_size', type=int, default=7, help='kernel_size in conv model')
    arg_parser.add_argument('--overlap_k', type=int, default=0, help='overlap in preprocess')
    arg_parser.add_argument('--batch_size', type=int, default=4)
    arg_parser.add_argument('--hw2_QA_bert', type=str, default='../bert.pth', help='path of hw2_QA_bert')
    arg_parser.add_argument('--merge_type', type=int, default=0, help='merge type')
    args = arg_parser.parse_args()
    return args

def main():
    args = Arg()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:%d" % args.cuda if use_cuda else "cpu")
    print('device = {}'.format(device), file = sys.stderr)
    BATCH_SIZE = args.batch_size
    # setup logging
    logging.basicConfig(
        level=logging.INFO, 
        format='%(message)s', 
        handlers=[logging.FileHandler(args.log_name, 'w'), logging.StreamHandler(sys.stdout)]
    )
    logging.info(args)
    
    # set up tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    
    # load data
    train_path = os.path.join(args.train_data, 'ca_data/*')
    dev_path = os.path.join(args.dev_data, 'ca_data/*')
    
    print('Load train data...', file=sys.stderr)
    QAExamples_train, train_index_list = preprocess_parent(train_path, tokenizer, args.overlap_k, (0 if args.overlap_k != 0 else 1), merge_type=args.merge_type)
    QAFeatures_train, QAdataset_train = QAExamples_to_QAFeatureDataset(QAExamples_train, tokenizer, 'train')
    print('DONE', file=sys.stderr)
    
    print('Load dev data...', file=sys.stderr)
    QAExamples_dev, dev_index_list = preprocess_parent(dev_path, tokenizer, 0, 1, merge_type=args.merge_type)
    QAFeatures_dev, QAdataset_dev = QAExamples_to_QAFeatureDataset(QAExamples_dev, tokenizer, 'train')
    print('DONE', file=sys.stderr)
    
    # train model
    if args.train != None:
        if args.use_sampler:
            sampler = WeightedRandomSampler(
                    [args.ratio if feature.start_position != 0 else 1 for feature in QAFeatures_train],
                    args.round * BATCH_SIZE,
                    replacement=True
            )
            train_dataloader = DataLoader(QAdataset_train, batch_size=BATCH_SIZE, sampler=sampler)
        else:
            train_dataloader = DataLoader(QAdataset_train, batch_size=BATCH_SIZE, shuffle=True)
        dev_dataloader = DataLoader(QAdataset_dev, batch_size=BATCH_SIZE, shuffle=False)
        
        model = Model(args.pretrained_model, model_type=args.train, kernel_size=args.kernel_size)
        if args.hw2_QA_bert is not None:
            model.model.load_state_dict(torch.load(args.hw2_QA_bert, map_location=device)['bert_state_dict'])

        optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
        criterion = nn.CrossEntropyLoss()
        Train(model, train_dataloader, dev_dataloader, dev_index_list, QAExamples_dev, QAFeatures_dev, tokenizer, criterion, optimizer, device, args.model_name, args.dev_ref_file, epochs=args.epochs)

if __name__ == '__main__':
    main()
