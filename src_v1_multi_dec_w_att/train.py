'''
Author: Aman
Date: 2021-11-15 10:41:17
Contact: cq335955781@gmail.com
LastEditors: Aman
LastEditTime: 2021-12-03 02:09:35
'''


import argparse
import logging
import math
import os
import random
import time
import pdb

import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, BertTokenizer, get_linear_schedule_with_warmup

from configs import model_cfgs, data_config
from model import EXPTeller
from MyDataset import MyDataset
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


parser = argparse.ArgumentParser()
parser.add_argument("--device_ids", default="[0,1,2,3]", type=str, help="GPU device ids")
parser.add_argument("--batch_size", default=256, type=int, help="Batch size")
parser.add_argument("--val_batch_size", default=256, type=int, help="Eval batch size")
parser.add_argument("--epochs", default=50, type=int, help="Number of epochs")
parser.add_argument("--lr", default=1e-02, type=float, help="Learning rate")
parser.add_argument("--teacher_forcing_ratio", default=0.8, type=float, help="Teacher forcing ratio")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--num_workers", default=8, type=int, help="Number of workers")
parser.add_argument("--log_interval", default=100, type=int, help="Log interval")
parser.add_argument("--val_interval_ratio", default=0.2, type=float, help="Eval once every interval ratio of training data")
parser.add_argument("--train_data_path", default="../datasets/sample_data/data_train_44000.pkl", type=str, help="Train data path")
parser.add_argument("--val_data_path", default="../datasets/sample_data/data_val_2000.pkl", type=str, help="Val data path")
parser.add_argument("--save_model", default=True, type=bool, help="Save model or not")
parser.add_argument("--save_path", default="./models/test1", type=str, help="Save directory")
# parser.add_argument("--save_interval", default=1, type=int, help="Save interval")
parser.add_argument("--log_path", default="../logs/test1.log", type=str, help="Log directory")


global args
args = parser.parse_args()
batch_size = args.batch_size
val_batch_size = args.val_batch_size
model_cfgs = model_cfgs
data_config = data_config()
print(args, model_cfgs)
logging.basicConfig(filename=args.log_path,
                    level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)-2s - %(filename)-8s : %(lineno)s line - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)
logger.info(args)


# load tokenizer
# ADD_TOKENS_LIST = ['[#START#]', '[#EOS#]']
# tokenizer = BertTokenizer.from_pretrained("bert-base-chinese", never_split=ADD_TOKENS_LIST)
# tokenizer.vocab['[#EOS#]'] = tokenizer.vocab.pop('[unused1]')
# tokenizer.vocab['[#START#]'] = tokenizer.vocab.pop('[unused2]')
# tokenizer.vocab['[#END#]'] = tokenizer.vocab.pop('[unused3]')

tokenizer = BertTokenizer.from_pretrained("./vocab/vocab_sm.txt")



devices = list(eval(args.device_ids))
multi_gpu = False
if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name())
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        multi_gpu = True
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy`` and ``torch``.
 
    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ^^ safe to call this function even if cuda is not available
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


print("Loading data...")
train_data_file = args.train_data_path
val_data_file = args.val_data_path
train_data = MyDataset(train_data_file, tokenizer, data_config)
valid_data = MyDataset(val_data_file, tokenizer, data_config)
print("Data loaded.")


def main(trial=None):
    if trial: # This is for optuna optimization of hyperparameters (learning rate, dropout rate, etc.)
        args.lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
        # args.dropout = trial.suggest_uniform("dropout", 0.0, 0.5)
        # args.epochs = trial.suggest_int("epochs", 5, 15)
        # model_cfgs['image']['num_layers'] = trial.suggest_int("image_rnn_num_layers", 2, 8)


    model = EXPTeller(model_cfgs, len(tokenizer.vocab))
    
    n_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print('* number of parameters: %d' % n_params)
    logger.info('* number of parameters: %d' % n_params)  # compute the number of parameters

    if multi_gpu:
        model = nn.DataParallel(model, device_ids = devices)
        # model.to(f'cuda:{model.device_ids[0]}')
        model.to(device)
    else:
        model = model.to(device)

    res = train(model, train_data, valid_data)

    return res


def train(model, train_data, valid_data):
    print("Now lr is ", args.lr)
    logger.info('Now lr is %s.' % args.lr)
    train_dataset = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    valid_dataset = DataLoader(valid_data, batch_size=val_batch_size, shuffle=True, num_workers=args.num_workers)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    training_steps = int(len(train_dataset) * args.epochs)
    print('Total training steps:', training_steps)
    logger.info('* number of training steps: %d' % training_steps) # number of training steps
    one_epoch_steps = len(train_dataset)
    # warmup and decay the learning rate
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = int(one_epoch_steps * 0.1), 
                                                num_training_steps = training_steps)
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        t1 = time.time()
        torch.cuda.empty_cache()
        print("\nEpoch ", epoch + 1, "/", args.epochs)
        logger.info("Epoch " + str(epoch + 1) + "/" + str(args.epochs))
        avg_loss = 0.0
        model.train()
        # Setting the tqdm progress bar
        epoch_iterator = tqdm(enumerate(train_dataset),
                              desc="%s: %d/%d Epochs >> Steps" % ("Train", epoch + 1, args.epochs),
                              total=len(train_dataset),
                              bar_format="{l_bar}{r_bar}")
        for step, batch in epoch_iterator:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model.forward(batch, args.teacher_forcing_ratio) # [batch_size, seq_len, output_len, vocab_size]
            outputs = outputs[:,:,1:,:].contiguous().view(-1, outputs.shape[-1])
            targets = batch['targets'][:,:,1:].contiguous().view(-1)
            loss = criterion(outputs, targets)
            loss = loss.mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # clip gradient
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            for param_group in optimizer.param_groups:
                args.lr = param_group['lr']
            epoch_iterator.set_postfix(lr=args.lr, ppl=math.exp(loss.item()), loss=loss.item())  # show the learning rate and loss on the progress bar 
            if step > 0 and (step + 1) % int(one_epoch_steps * args.val_interval_ratio) == 0:
                val_loss, ppl = evaluate(model, valid_dataset, criterion)
                logger.info("Epoch: %d, Step: %d/%d, Val. Loss: %.4f, Val. PPL: %.3f" % (epoch + 1, step + 1, one_epoch_steps, val_loss, ppl))
                print(" Epoch: %d, Step: %d/%d, Val. Loss: %.4f, Val. PPL: %.3f" % (epoch + 1, step + 1, one_epoch_steps, val_loss, ppl))
                # Save model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if args.save_model:
                        if not os.path.exists(args.save_path):
                            os.makedirs(args.save_path)
                        state = {'model': model.state_dict(), 'args': args, 'model_cfgs': model_cfgs} # 'optimizer': optimizer.state_dict(), 
                        torch.save(state, args.save_path + "/model.pth")
                        logger.info("Epoch: %d, Step: %d, Saving Model to \'%s\'." % (epoch + 1, step, args.save_path))
                        print("Epoch: %d, Step: %d, Saving Model to \'%s\'." % (epoch + 1, step, args.save_path))
                model.train()
            avg_loss += loss.item()
            if step > 0 and (step + 1) % args.log_interval == 0:
                logger.info("Epoch: %d, Step: %d/%d, Average loss: %.6f" % (epoch + 1, step + 1, one_epoch_steps, avg_loss / (step + 1)))
        # End of epoch
        val_loss, ppl = evaluate(model, valid_dataset, criterion)
        logger.info("End eval of epoch %d. Val. Loss: %.4f, Val. PPL: %.2f" % (epoch + 1, val_loss, ppl))
        print("End eval of epoch %d. Val. Loss: %.4f, Val. PPL: %.2f" % (epoch + 1, val_loss, ppl))
        model.train()
        logger.info("Average loss: %.4f  Elapsed time: %s" % (avg_loss / (one_epoch_steps + 1), format_time(time.time()-t1)))
        print("Average loss: %.4f  Elapsed time: %s" % (avg_loss / (one_epoch_steps + 1), format_time(time.time()-t1)))
    logger.info("Training finished.")
    print("Training finished.")

    return ppl


def evaluate(model, valid_dataset, criterion):
    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        # epoch_iterator = tqdm(valid_dataset, ncols=200, leave=False)
        for i, batch in enumerate(valid_dataset):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model.forward(batch, 0) # [batch_size, seq_len, output_len, vocab_size]
            outputs = outputs[:,:,1:,:].contiguous().view(-1, outputs.shape[-1])
            targets = batch['targets'][:,:,1:].contiguous().view(-1)
            loss = criterion(outputs, targets)
            loss = loss.mean()
            valid_loss += loss.item()
    valid_loss /= len(valid_dataset)
    ppl = math.exp(valid_loss)

    return valid_loss, ppl


def optuna_optimize(trial_times):
    study = optuna.create_study(direction="minimize")
    study.optimize(main, n_trials=trial_times)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":

    time_begin = time.time()
    set_seed(args.seed)

    main()

    # optuna_optimize(25)

    time_end = time.time()
    print("Finished!\nTotal time: %s" % format_time(time_end - time_begin))
    











