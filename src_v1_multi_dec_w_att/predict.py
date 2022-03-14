'''
Author: Aman
Date: 2021-11-19 00:39:07
Contact: cq335955781@gmail.com
LastEditors: Aman
LastEditTime: 2021-12-03 09:01:37
'''

import argparse
import math
import os
import pdb

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer

from configs import model_cfgs, data_config
from model import EXPTeller
from MyDataset import MyDataset
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

parser = argparse.ArgumentParser()
parser.add_argument("--device_ids", default="[0,1,2,3]", type=str, help="GPU device ids")
parser.add_argument("--test_batch_size", default=128, type=int, help="Test batch size")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--num_workers", default=4, type=int, help="Number of workers")
parser.add_argument("--data_path", default="../datasets/sample_data/data_test_283.pkl", type=str, help="Data directory")
parser.add_argument("--model_path", default="./models/test1/model.pth", type=str, help="Model path")

global args
args = parser.parse_args()
model_cfgs = model_cfgs
model_cfgs['teacher_forcing_ratio'] = 0
data_config = data_config()
print(args, model_cfgs)

devices = list(eval(args.device_ids))
device = torch.device("cuda")


# load tokenizer
# ADD_TOKENS_LIST = ['[#START#]', '[#EOS#]']
# tokenizer = BertTokenizer.from_pretrained("bert-base-chinese", never_split=ADD_TOKENS_LIST)
# tokenizer.vocab['[#EOS#]'] = tokenizer.vocab.pop('[unused1]')
# tokenizer.vocab['[#START#]'] = tokenizer.vocab.pop('[unused2]')
# self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
# tokenizer.vocab['[#END#]'] = tokenizer.vocab.pop('[unused3]')
# sent = "[#START#]我心态崩了[EOS]哎！！！！！[SEP]"
# print(tokenizer.tokenize(sent))
# print(tokenizer.vocab)
# exit()
tokenizer = BertTokenizer.from_pretrained("./vocab/vocab_sm.txt")
print("vocab_size: ", len(tokenizer.vocab))

# load model
checkpoint = torch.load(args.model_path)
_args = checkpoint['args']
model = EXPTeller(model_cfgs, len(tokenizer.vocab))
model.to(device)
model = nn.DataParallel(model, device_ids=devices)
model.load_state_dict(checkpoint['model'])


criterion = nn.CrossEntropyLoss()
def predict(model, test_dataset):
    model.eval()
    test_loss = 0.0
    y_pred = []
    y_label = []
    with torch.no_grad():
        epoch_iterator = tqdm(test_dataset, ncols=100, leave=False)
        for i, batch in enumerate(epoch_iterator):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model.forward(batch) # [batch_size, seq_len, output_len, vocab_size]
            y_pred.append(torch.argmax(outputs, dim=3))
            y_label.append(batch['targets'])
            outputs = outputs[:,:,1:,:].contiguous().view(-1, outputs.shape[-1])
            targets = batch['targets'][:,:,1:].contiguous().view(-1)
            loss = criterion(outputs, targets)
            loss = loss.mean()
            test_loss += loss.item()
    y_pred = torch.cat(y_pred, dim=0)
    y_label = torch.cat(y_label, dim=0)
    test_loss /= len(test_dataset)
    ppl = math.exp(test_loss)

    return test_loss, ppl, y_pred, y_label


def display_i(idx, y_pred, y_label):
    # display the idx-th result
    print("\n")
    print("="*100)
    print("idx:", idx)
    # y_pred = torch.cat(y_pred)
    # y_label = torch.cat(y_label)
    y_pred = y_pred[idx].cpu().numpy()
    y_label = y_label[idx].cpu().numpy()
    y_pred = [tokenizer.convert_ids_to_tokens(line) for line in y_pred]
    y_label = [tokenizer.convert_ids_to_tokens(line) for line in y_label]
    print("Prediction:\n")
    for line in y_pred:
        print(line)
    print("-"*80)
    print("Label:\n")
    for line in y_label:
        print(line)
    print("="*100)


if __name__ == "__main__":
    print("Loading data...")
    test_data_file = args.data_path
    test_data = MyDataset(test_data_file, tokenizer, data_config)
    test_dataset = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers)
    print("Data test loaded.")
    test_loss, ppl, y_pred, y_label = predict(model, test_dataset)
    # import pdb; pdb.set_trace()
    print("Test loss: %.4f, PPL: %.4f" % (test_loss, ppl))
    print("Now displaying any instance of the test data. 0 <= idx < %d" % (len(test_data)))
    while True:
        # input a idx
        i = eval(input("Input a idx: "))
        display_i(i, y_pred, y_label)


