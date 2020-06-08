import torch
import argparse
from transformers import BertTokenizer, BertModel, AdamW

from model import Model

def Arg():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--train', type = str, help = 'conv for convolution, linear for linear')
    arg_parser.add_argument('--cuda', default = 0, type = int, help = 'cuda device number')
    args = arg_parser.parse_args()
    return args

def main():
    args = Arg()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:%d" % args.cuda if use_cuda else "cpu")
    print('device = {}'.format(device))
    BATCH_SIZE = 32

    if args.train != None:
        # TODO: train_dataloader, dev_dataloader
        pretrained_model = BertModel.from_pretrained('bert-base-multilingual-cased')
        model = Model(pretrained_model, convolution = (args.train == 'conv'))

        optimizer = AdamW(model.parameters(), lr = 2e-6, eps = 1e-8)
        criterion = nn.CrossEntropyLoss()
        Train(model, train_dataloader, dev_dataloader, criterion, optimizer, device)

if __name__ == '__main__':
    main()
