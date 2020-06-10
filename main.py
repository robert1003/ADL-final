import os
import sys
import argparse
import torch
import torch.nn as nn
import logging
from transformers import AutoTokenizer, BertModel, AdamW
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import tqdm
from model import Model
from _QA import QAExamples_to_QAFeatureDataset
from preprocess import preprocess
from train import Train

def Arg():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--train', type=str, help='conv for convolution, linear for linear')
    arg_parser.add_argument('--cuda', default=0, type=int, help='cuda device number')
    arg_parser.add_argument('--model_name', type=str, required=True, help='model save name')
    arg_parser.add_argument('--log_name', type=str, required=True, help='log file name')
    arg_parser.add_argument('--train_data', type=str, default='./release/train', help='train data folder')
    arg_parser.add_argument('--dev_data', type=str, default='./release/dev', help='dev data folder')
    arg_parser.add_argument('--pretrained_model', type=str, default='bert-base-multilingual-cased', help='name of pretrained model to use')
    arg_parser.add_argument('--epochs', type=int, default=10, help='training epochs')
    arg_parser.add_argument('--learning_rate', type=float, default=3e-6, help='learning rate')
    arg_parser.add_argument('--use_sampler', action='store_true', help='use sampler to solve imbalance problem')
    arg_parser.add_argument('--ratio', type=float, help='use sampler to solve imbalance problem, \
            sum(prob(answerable)) / sum(prob(unanswerable)) what you enter')
    arg_parser.add_argument('--kernel_size', type=int, default=3, help='kernel_size in conv model')
    args = arg_parser.parse_args()
    return args

def main():
    args = Arg()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:%d" % args.cuda if use_cuda else "cpu")
    print('device = {}'.format(device), file = sys.stderr)
    BATCH_SIZE = 4
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
    QAExamples_train, train_index_list = preprocess(train_path, tokenizer, 0, 1)
    QAFeatures_train, QAdataset_train = QAExamples_to_QAFeatureDataset(QAExamples_train, tokenizer, 'train')
    print('DONE', file=sys.stderr)
    
    print('Load dev data...', file=sys.stderr)
    QAExamples_dev, dev_index_list = preprocess(dev_path, tokenizer, 0, 1)
    QAFeatures_dev, QAdataset_dev = QAExamples_to_QAFeatureDataset(QAExamples_dev, tokenizer, 'train')
    print('DONE', file=sys.stderr)
    
    # data checking
    print('Checking data...', file=sys.stderr)
    for fea in tqdm(QAFeatures_train):
        data = QAExamples_train[fea.example_id]
        s = tokenizer.decode(fea.input_ids[fea.start_position + 15:fea.end_position + 15 + 1], skip_special_tokens=False).replace(' ', '').replace('#', '')
        try:
            assert data.answer_text.find(s) != -1 or (s == '[SEP]' and not data.answerable)
        except:
            if s.find('[UNK]') == -1:
                print(fea.tokens[fea.start_position:fea.end_position + 1], file=sys.stderr)
                print(s, file=sys.stderr)
                print(len(data.answer_text), file=sys.stderr)
                print(data, file=sys.stderr)
                print('\n\n', file=sys.stderr)

        if fea.start_position >= 497 or fea.end_position >= 497:
            print('fffffffff', fea.tokens[fea.start_position + 15:fea.end_position + 15 + 1], sys.stderr)
        
    for fea in tqdm(QAFeatures_dev):
        data = QAExamples_dev[fea.example_id]
        s = tokenizer.decode(fea.input_ids[fea.start_position + 15:fea.end_position + 15 + 1], skip_special_tokens=False).replace(' ', '').replace('#', '')
        try:
            assert data.answer_text.find(s) != -1 or (s == '[SEP]' and not data.answerable)
        except:
            if s.find('[UNK]') == -1:
                print(fea.tokens[fea.start_position:fea.end_position + 1], file=sys.stderr)
                print(s, file=sys.stderr)
                print(len(data.answer_text), file=sys.stderr)
                print(data, file=sys.stderr)
                print('\n\n', file=sys.stderr)

        if fea.start_position >= 497 or fea.end_position >= 497:
            print('fffffffff', fea.tokens[fea.start_position + 15:fea.end_position + 15 + 1], file=sys.stderr)
    print('DONE', file=sys.stderr)
    
    # train model
    if args.train != None:
        if args.use_sampler:
            answerable_cnt = sum([1 if feature.start_position != 0 else 0 for feature in QAFeatures_train])
            answerable_weight = (len(QAFeatures_train) - answerable_cnt) / answerable_cnt
            sampler = WeightedRandomSampler(
                    [answerable_weight * args.ratio if feature.start_position != 0 else 1 for feature in QAFeatures_train],
                    len(QAFeatures_train),
                    replacement=True
            )
            train_dataloader = DataLoader(QAdataset_train, batch_size=BATCH_SIZE, sampler=sampler)
        else:
            train_dataloader = DataLoader(QAdataset_train, batch_size=BATCH_SIZE, shuffle=True)
        dev_dataloader = DataLoader(QAdataset_dev, batch_size=BATCH_SIZE, shuffle=False)
        
        pretrained_model = BertModel.from_pretrained(args.pretrained_model)
        model = Model(pretrained_model, convolution=args.train, kernel_size=args.kernel_size)

        optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
        criterion = nn.CrossEntropyLoss()
        Train(model, train_dataloader, dev_dataloader, criterion, optimizer, device, args.model_name, epochs=args.epochs)

if __name__ == '__main__':
    main()
