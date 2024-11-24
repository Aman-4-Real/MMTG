'''
Author: Aman
Date: 2022-03-21 19:38:25
Contact: cq335955781@gmail.com
LastEditors: Aman
LastEditTime: 2022-08-10 18:41:31
'''


import argparse
import logging
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, BertTokenizer, get_linear_schedule_with_warmup

from configs import model_cfgs, data_config
from model import MMTG
from MyDataset import MyDataset
from utils import *
from loss import MyLoss

os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"


parser = argparse.ArgumentParser()
parser.add_argument("--device_ids", default="0", type=str, help="GPU device ids")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
parser.add_argument("--val_batch_size", default=32, type=int, help="Eval batch size")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs")
parser.add_argument("--lr", default=1e-05, type=float, help="Learning rate")
parser.add_argument("--curriculums", default="[1,3]", type=str, help="Curriculum rate")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--num_workers", default=0, type=int, help="Number of workers")
parser.add_argument("--log_interval", default=100, type=int, help="Log interval")
parser.add_argument("--val_interval_ratio", default=0.2, type=float, help="Eval once every interval ratio of training data")
parser.add_argument("--train_data_path", default="", type=str, help="Train data path")
parser.add_argument("--val_data_path", default="", type=str, help="Val data path")
parser.add_argument("--save_model", action='store_true', help="Save model")
parser.add_argument("--save_path", default="", type=str, help="Save directory")
parser.add_argument("--log_path", default="", type=str, help="Log directory")
parser.add_argument("--alpha", default=0, type=float, help="Factor of KLDivLoss.")

args = parser.parse_args()
batch_size = args.batch_size
val_batch_size = args.val_batch_size
curriculums = eval(args.curriculums)
model_cfgs = model_cfgs
data_config = data_config()
print(args, model_cfgs)
logging.basicConfig(filename=args.log_path,
                    level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)-2s - %(filename)-8s : %(lineno)s line - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)
logger.info(args)
import pdb; pdb.set_trace()
tokenizer = BertTokenizer.from_pretrained("./vocab/vocab.txt")

devices = eval('['+args.device_ids+']')
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def main():

    print("Loading data...")
    train_data_file = args.train_data_path
    val_data_file = args.val_data_path
    train_data = MyDataset(train_data_file, tokenizer, data_config)
    valid_data = MyDataset(val_data_file, tokenizer, data_config)
    print("Data loaded.")

    model = MMTG(model_cfgs, data_config, len(tokenizer.vocab), train_flag=True)
    
    n_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print('* number of parameters: %d' % n_params)
    logger.info('* number of parameters: %d' % n_params)  # compute the number of parameters

    if multi_gpu:
        model = nn.DataParallel(model, device_ids=devices)
        model.to(device)
    else:
        model = model.to(device)

    res = train(model, train_data, valid_data)

    return res


def train(model, train_data, valid_data):

    print("Now lr is ", args.lr, "Now batch_size is ", args.batch_size)
    logger.info('Now lr is %s, batch_size is %s.' % (args.lr, args.batch_size))
    
    ### This is for the different numbers of samples in different curriculum stages
    ### The following is a simple but inefficient way to solve this. You can definitely change it in your own way to save more memeory resource.
    train_dataset_1 = DataLoader(train_data, batch_size=2*batch_size, shuffle=True, num_workers=args.num_workers)
    valid_dataset_1 = DataLoader(valid_data, batch_size=2*val_batch_size, shuffle=True, num_workers=args.num_workers)
    train_dataset_2 = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    valid_dataset_2 = DataLoader(valid_data, batch_size=val_batch_size, shuffle=True, num_workers=args.num_workers)
    train_datasets = [train_dataset_1, train_dataset_2, train_dataset_2]
    valid_datasets = [valid_dataset_1, valid_dataset_2, valid_dataset_2]

    optimizer = AdamW(model.parameters(), lr=args.lr)
    training_steps = int(len(train_dataset_1) * curriculums[0] + \
                         len(train_dataset_2) * (curriculums[1] - curriculums[0]) + \
                         len(train_dataset_2) * (args.epochs - curriculums[1]))
    print('Total training steps:', training_steps)
    logger.info('* number of training steps: %d' % training_steps) # number of training steps
    one_epoch_steps = len(train_dataset_1)

    # warmup and decay the learning rate
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = int(one_epoch_steps * 0.1), 
                                                num_training_steps = training_steps)
                                                
    criterion = MyLoss(data_config, model_cfgs)
    best_val_loss = float("inf")
    global_steps = 0
    stage = 0 # curriculum stage
    for epoch in range(args.epochs):
        t1 = time.time()
        torch.cuda.empty_cache()
        print("\nEpoch ", epoch + 1, "/", args.epochs)
        logger.info("Epoch " + str(epoch + 1) + "/" + str(args.epochs))
        if epoch < curriculums[0]: # very positive and negative first
            stage = 1
            epoch_iterator = tqdm(enumerate(train_dataset_1),
                                    desc="%s: %d/%d Epochs >> Steps" % ("Train", epoch + 1, args.epochs),
                                    total=len(train_dataset_1),
                                    bar_format="{l_bar}{r_bar}")
        else:
            if epoch < curriculums[1]: # then positive and negative
                stage = 2
            else:
                stage = 3
            epoch_iterator = tqdm(enumerate(train_dataset_2),
                                    desc="%s: %d/%d Epochs >> Steps" % ("Train", epoch + 1, args.epochs),
                                    total=len(train_dataset_2),
                                    bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0
        model.train()
        for step, batch in epoch_iterator:
            if stage == 1:
                idxs = torch.cat([torch.where(batch['rating']<2)[0], torch.where(batch['rating']>4)[0]])
            elif stage == 2:
                idxs = torch.cat([torch.where(batch['rating']<3)[0], torch.where(batch['rating']>3)[0]])
            else:
                idxs = torch.arange(len(batch['rating']))
            if len(idxs) == 0:
                continue
            batch = {k: v[idxs].to(device) for k, v in batch.items()}
            ratings = batch['rating'].to(device)
            _loss, kl_loss, outputs = model.forward(batch)
            outputs = outputs.contiguous()
            targets = batch['targets'].contiguous()
            loss = criterion(outputs, targets, ratings, stage)
            total_loss = loss.mean() + args.alpha * kl_loss.mean()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # clip gradient
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            for param_group in optimizer.param_groups:
                args.lr = param_group['lr']
            epoch_iterator.set_postfix(lr=args.lr, loss=total_loss.item())  # show the learning rate and loss on the progress bar
            global_steps += 1
            if step > 0 and (step + 1) % int(len(train_datasets[stage-1]) * args.val_interval_ratio) == 0:
                val_loss, _ = evaluate(model, valid_datasets[stage-1], stage, criterion)
                logger.info("Epoch: %d, Step: %d/%d, Val. Loss: %.4f" % (epoch + 1, step + 1, len(train_datasets[stage-1]), val_loss))
                print(" Epoch: %d, Step: %d/%d, Val. Loss: %.4f" % (epoch + 1, step + 1, len(train_datasets[stage-1]), val_loss))
                # Save model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if args.save_model:
                        if not os.path.exists(args.save_path):
                            os.makedirs(args.save_path)
                        state = {'model': model.state_dict(), 'args': args, 'model_cfgs': model_cfgs}
                        torch.save(state, args.save_path + f"/best_val_model.pth")
                        logger.info("Epoch: %d, Step: %d, Saving Model to \'%s\'." % (epoch + 1, step, args.save_path))
                        print("Epoch: %d, Step: %d, Saving Model to \'%s\'." % (epoch + 1, step, args.save_path))
                model.train()
            avg_loss += loss.item()
            if step > 0 and (step + 1) % args.log_interval == 0:
                logger.info("Epoch: %d, Step: %d/%d, Average loss: %.6f" % (epoch + 1, step + 1, len(train_datasets[stage-1]), avg_loss / (step + 1)))
        # End of epoch
        val_loss, _ = evaluate(model, valid_datasets[stage-1], stage, criterion)
        logger.info("End eval of epoch %d. Val. Loss: %.4f" % (epoch + 1, val_loss))
        print("End eval of epoch %d. Val. Loss: %.4f" % (epoch + 1, val_loss))
        model.train()
        logger.info("Average loss: %.4f  Elapsed time: %s" % (avg_loss / (len(train_datasets[stage-1]) + 1), format_time(time.time()-t1)))
        print("Average loss: %.4f  Elapsed time: %s" % (avg_loss / (len(train_datasets[stage-1]) + 1), format_time(time.time()-t1)))
        if args.save_model:
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            state = {'model': model.state_dict(), 'args': args, 'model_cfgs': model_cfgs}
            torch.save(state, args.save_path + f"/epoch_{epoch + 1}.pth")
            logger.info("Epoch: %d, Step: %d, Saving Model to \'%s\'." % (epoch + 1, step, args.save_path))
            print("Epoch: %d, Step: %d, Saving Model to \'%s\'." % (epoch + 1, step, args.save_path))
    
    logger.info("Training finished.")
    print("Training finished.")

    return val_loss


def evaluate(model, valid_dataset, stage, criterion):
    model.eval()
    valid_loss = 0.0
    kldiv_loss = 0.0
    with torch.no_grad():
        epoch_iterator = tqdm(valid_dataset, ncols=100, leave=False)
        for i, batch in enumerate(epoch_iterator):
            if stage == 1:
                idxs = torch.cat([torch.where(batch['rating']<2)[0], torch.where(batch['rating']>4)[0]])
            elif stage == 2:
                idxs = torch.cat([torch.where(batch['rating']<3)[0], torch.where(batch['rating']>3)[0]])
            else:
                idxs = torch.arange(len(batch['rating']))
            if len(idxs) == 0:
                continue
            batch = {k: v[idxs].to(device) for k, v in batch.items()}
            ratings = batch['rating'].to(device)
            _loss, kl_loss, outputs = model.forward(batch)
            outputs = outputs.contiguous()
            targets = batch['targets'].contiguous()
            loss = criterion(outputs, targets, ratings, stage)          
            total_loss = loss.mean() + args.alpha * kl_loss.mean()
            valid_loss += total_loss.item()
            kldiv_loss += args.alpha * kl_loss.mean().item()
    valid_loss /= len(valid_dataset)
    kldiv_loss /= len(valid_dataset)

    return valid_loss, kldiv_loss



if __name__ == "__main__":

    time_begin = time.time()
    set_seed(args.seed)

    main()
    
    time_end = time.time()
    print("Finished!\nTotal time: %s" % format_time(time_end - time_begin))
    


