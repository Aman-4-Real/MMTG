'''
Author: Aman
Date: 2022-01-13 01:00:15
Contact: cq335955781@gmail.com
LastEditors: Aman
LastEditTime: 2022-01-21 23:41:29
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
from torch.utils.tensorboard import SummaryWriter   

from configs import model_cfgs, data_config
from model_wo_img import EXPTeller
from MyDataset import MyDataset
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


parser = argparse.ArgumentParser()
parser.add_argument("--device_ids", default="[0,1,2,3]", type=str, help="GPU device ids")
parser.add_argument("--batch_size", default=96, type=int, help="Batch size")
parser.add_argument("--val_batch_size", default=128, type=int, help="Eval batch size")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs")
parser.add_argument("--lr", default=1e-05, type=float, help="Learning rate")
parser.add_argument("--curriculums", default=[1,3], type=float, help="Curriculum rate")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--num_workers", default=8, type=int, help="Number of workers")
parser.add_argument("--log_interval", default=100, type=int, help="Log interval")
parser.add_argument("--val_interval_ratio", default=0.2, type=float, help="Eval once every interval ratio of training data")
parser.add_argument("--train_data_path", default="../datasets/new_data_rating/train_data_with_ratings_210k.pkl", type=str, help="Train data path")
parser.add_argument("--val_data_path", default="../datasets/new_data_rating/val_data_with_ratings_8k.pkl", type=str, help="Val data path")
parser.add_argument("--save_model", default=True, type=bool, help="Save model or not")
parser.add_argument("--save_path", default="./models/final_wo_img_5ep_1e-5_bsz96_cl_ln", type=str, help="Save directory")
# parser.add_argument("--save_interval", default=1, type=int, help="Save interval")
parser.add_argument("--log_path", default="./logs/final_wo_img_5ep_1e-5_bsz96_cl_ln.log", type=str, help="Log directory")
parser.add_argument("--tensorboard_log_dir", default="./logs/final_wo_img_5ep_1e-5_bsz96_cl_ln", type=str, help="Tensorboard log directory")

global args
args = parser.parse_args()
batch_size = args.batch_size
val_batch_size = args.val_batch_size
curriculums = list(args.curriculums)
model_cfgs = model_cfgs
data_config = data_config()
print(args, model_cfgs)
logging.basicConfig(filename=args.log_path,
                    level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)-2s - %(filename)-8s : %(lineno)s line - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)
logger.info(args)
if not os.path.exists(args.tensorboard_log_dir):
    os.makedirs(args.tensorboard_log_dir)
writer = SummaryWriter(args.tensorboard_log_dir)

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
        # args.lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
        args.lr = trial.suggest_discrete_uniform("lr", 5e-5, 1e-3, 5e-5)
        # args.dropout = trial.suggest_uniform("dropout", 0.0, 0.5)
        # args.epochs = trial.suggest_int("epochs", 5, 15)
        # model_cfgs['image']['num_layers'] = trial.suggest_int("image_rnn_num_layers", 2, 8)


    model = EXPTeller(model_cfgs, len(tokenizer.vocab), True)
    
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


class MyNLLLoss(torch.nn.Module):
    def __init__(self):
        super(MyNLLLoss, self).__init__()

    def nll(self, y, p):
        NEAR_0 = 1e-10
        p = torch.exp(torch.sum(torch.log(p)))
        return - y * torch.log(p + NEAR_0) - (1 - y) * torch.log(1 - p + NEAR_0)

    def forward(self, preds, targets, ratings, stage):
        '''
        Args:
            preds: (batch_size, seq_len, _max_sent_length*2, vocab_size)
            targets: (batch_size, seq_len, _max_sent_length*2)
            ratings: (batch_size)
        '''
        device = preds.device
        zero = torch.zeros_like(ratings)
        one = torch.ones_like(ratings)
        if stage == 1:
            ratings = torch.where(ratings > 4, one, zero)
        else:
            ratings = torch.where(ratings > 3, one, zero)
        # (n, 5, 40, 24408)
        # preds = preds.view(preds.shape[0], -1, preds.shape[-1])
        # targets = targets.view(targets.shape[0], -1)
        preds = nn.Softmax(dim=-1)(preds)
        loss = torch.zeros(preds.shape[0]).to(device)
        for i in range(preds.shape[0]): # batch_size
            y = ratings[i]
            for j in range(preds.shape[1]): # seq_len
                token_probs = preds[i,j,torch.arange(preds.shape[2]),targets[i][j]]
                # loss[i] += self.nll(y, torch.prod(token_probs))
                loss[i] += self.nll(y, token_probs)
        
        return torch.mean(loss/preds.shape[1])


class MyLoss(torch.nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        self._max_topic_len = data_config.topic_prompt_length
        self._seq_len = model_cfgs['seq_len']

    def forward(self, outputs, targets, ratings, stage):
        '''
        Args:
            outputs: (batch_size, topic_prompt_length + seq_len + _max_seq_length, vocab_size)
            targets: (batch_size, _max_seq_length)
            ratings: (batch_size)
        '''
        # import pdb; pdb.set_trace()
        NEAR_0 = 1e-10
        device = outputs.device
        zero = torch.zeros_like(ratings)
        one = torch.ones_like(ratings)
        batch_size = targets.shape[0]
        if stage == 1:
            ratings = torch.where(ratings > 4, one, zero)
        else:
            ratings = torch.where(ratings > 3, one, zero)

        shift_logits = outputs[:, self._max_topic_len+self._seq_len:-1, :]
        shift_labels = targets[:, 1:]
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss()
        
        loss = torch.zeros(batch_size).to(device)
        for i in range(batch_size):
            y = ratings[i]
            _loss = loss_fct(shift_logits[i], shift_labels[i])
            p = 1/torch.exp(_loss)
            # import pdb; pdb.set_trace()
            loss[i] += torch.sum(- y * torch.log(p + NEAR_0) - (1 - y) * torch.log(1 - p + NEAR_0))
        # import pdb; pdb.set_trace()
        return torch.mean(loss)
         


def train(model, train_data, valid_data):
    print("Now lr is ", args.lr)
    logger.info('Now lr is %s.' % args.lr)
    
    train_dataset_1 = DataLoader(train_data, batch_size=2*batch_size, shuffle=False, num_workers=args.num_workers)
    valid_dataset_1 = DataLoader(valid_data, batch_size=2*val_batch_size, shuffle=False, num_workers=args.num_workers)
    train_dataset_2 = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
    valid_dataset_2 = DataLoader(valid_data, batch_size=val_batch_size, shuffle=False, num_workers=args.num_workers)
    train_dataset_3 = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
    valid_dataset_3 = DataLoader(valid_data, batch_size=val_batch_size, shuffle=False, num_workers=args.num_workers)

    train_datasets = [train_dataset_1, train_dataset_2, train_dataset_3]
    valid_datasets = [valid_dataset_1, valid_dataset_2, valid_dataset_3]
    optimizer = AdamW(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    training_steps = int(len(train_dataset_1) * curriculums[0] + \
                         len(train_dataset_2) * (curriculums[1] - curriculums[0]) + \
                         len(train_dataset_3) * (args.epochs - curriculums[1]))
    print('Total training steps:', training_steps)
    logger.info('* number of training steps: %d' % training_steps) # number of training steps
    one_epoch_steps = len(train_dataset_1)
    # warmup and decay the learning rate
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = int(one_epoch_steps * 0.1), 
                                                num_training_steps = training_steps)
    # criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.CrossEntropyLoss()
    criterion = MyLoss()
    best_val_loss = float("inf")
    global_steps = 0
    stage = 0 # curriculum stage
    # graph_writed = 0 # whether the graph has been writed
    for epoch in range(args.epochs):
        t1 = time.time()
        torch.cuda.empty_cache()
        print("\nEpoch ", epoch + 1, "/", args.epochs)
        logger.info("Epoch " + str(epoch + 1) + "/" + str(args.epochs))
        if epoch < curriculums[0]: # hard positive and negative first
            stage = 1
            epoch_iterator = tqdm(enumerate(train_dataset_1),
                                    desc="%s: %d/%d Epochs >> Steps" % ("Train", epoch + 1, args.epochs),
                                    total=len(train_dataset_1),
                                    bar_format="{l_bar}{r_bar}")
        elif epoch < curriculums[1]: # then soft positive and negative
            stage = 2
            epoch_iterator = tqdm(enumerate(train_dataset_2),
                                    desc="%s: %d/%d Epochs >> Steps" % ("Train", epoch + 1, args.epochs),
                                    total=len(train_dataset_2),
                                    bar_format="{l_bar}{r_bar}")
        else:
            stage = 3
            epoch_iterator = tqdm(enumerate(train_dataset_3),
                                    desc="%s: %d/%d Epochs >> Steps" % ("Train", epoch + 1, args.epochs),
                                    total=len(train_dataset_3),
                                    bar_format="{l_bar}{r_bar}")
        avg_loss = 0.0
        model.train()
        # Setting the tqdm progress bar
        for step, batch in epoch_iterator:
            if stage == 1:
                idxs = torch.cat([torch.where(batch['rating']<2)[0], torch.where(batch['rating']>4)[0]])
            elif stage == 2:
                idxs = torch.cat([torch.where(batch['rating']<3)[0], torch.where(batch['rating']>3)[0]])
            else:
                idxs = torch.arange(len(batch['rating']))
            batch = {k: v[idxs].to(device) for k, v in batch.items()}
            ratings = batch['rating'].to(device)
            _loss, outputs = model.forward(batch) # [batch_size, seq_len+_max_seq_length, vocab_size]
            # import pdb; pdb.set_trace()
            outputs = outputs.contiguous() # .view(-1, outputs.shape[-1])
            targets = batch['targets'].contiguous() # .view(-1) # [batch_size, _max_seq_length]
            ppl_loss = torch.mean(_loss.mean(dim=-1))
            loss = criterion(outputs, targets, ratings, stage)
            # loss = criterion(outputs, targets)
            loss = loss.mean()
            # print("---loss: ", loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # clip gradient
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            for param_group in optimizer.param_groups:
                args.lr = param_group['lr']
            epoch_iterator.set_postfix(lr=args.lr, ppl=math.exp(ppl_loss), loss=loss.item())  # show the learning rate and loss on the progress bar
            global_steps += 1
            writer.add_scalar('train/loss', loss.item(), global_steps)
            writer.add_scalar('train/ppl_loss', ppl_loss, global_steps)
            writer.add_scalar('train/ppl', math.exp(ppl_loss), global_steps)
            # if not graph_writed:
            #     writer.add_graph(model.module, batch)
            #     graph_writed = 1
            if step > 0 and (step + 1) % int(len(train_datasets[stage-1]) * args.val_interval_ratio) == 0:
                val_loss, ppl_loss, ppl = evaluate(model, valid_datasets[stage-1], stage, criterion)
                logger.info("Epoch: %d, Step: %d/%d, Val. Loss: %.4f, Val. PPL: %.3f" % (epoch + 1, step + 1, len(train_datasets[stage-1]), val_loss, ppl))
                print(" Epoch: %d, Step: %d/%d, Val. Loss: %.4f, Val. PPL: %.3f" % (epoch + 1, step + 1, len(train_datasets[stage-1]), val_loss, ppl))
                writer.add_scalar('eval/loss', val_loss, global_steps)
                writer.add_scalar('eval/ppl_loss', ppl_loss, global_steps)
                writer.add_scalar('eval/ppl', ppl, global_steps)
                # Save model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if args.save_model:
                        if not os.path.exists(args.save_path):
                            os.makedirs(args.save_path)
                        state = {'model': model.state_dict(), 'args': args, 'model_cfgs': model_cfgs} # 'optimizer': optimizer.state_dict(), 
                        torch.save(state, args.save_path + f"/best_val_loss_{best_val_loss:.3f}.pth")
                        logger.info("Epoch: %d, Step: %d, Saving Model to \'%s\'." % (epoch + 1, step, args.save_path))
                        print("Epoch: %d, Step: %d, Saving Model to \'%s\'." % (epoch + 1, step, args.save_path))
                model.train()
            avg_loss += loss.item()
            if step > 0 and (step + 1) % args.log_interval == 0:
                logger.info("Epoch: %d, Step: %d/%d, Average loss: %.6f" % (epoch + 1, step + 1, len(train_datasets[stage-1]), avg_loss / (step + 1)))
        # End of epoch
        val_loss, ppl_loss, ppl = evaluate(model, valid_datasets[stage-1], stage, criterion)
        logger.info("End eval of epoch %d. Val. Loss: %.4f, Val. PPL: %.2f" % (epoch + 1, val_loss, ppl))
        print("End eval of epoch %d. Val. Loss: %.4f, Val. PPL: %.2f" % (epoch + 1, val_loss, ppl))
        writer.add_scalar('eval/loss', val_loss, global_steps)
        writer.add_scalar('eval/ppl_loss', ppl_loss, global_steps)
        writer.add_scalar('eval/ppl', ppl, global_steps)
        model.train()
        logger.info("Average loss: %.4f  Elapsed time: %s" % (avg_loss / (len(train_datasets[stage-1]) + 1), format_time(time.time()-t1)))
        print("Average loss: %.4f  Elapsed time: %s" % (avg_loss / (len(train_datasets[stage-1]) + 1), format_time(time.time()-t1)))
        if args.save_model:
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            state = {'model': model.state_dict(), 'args': args, 'model_cfgs': model_cfgs} # 'optimizer': optimizer.state_dict(), 
            torch.save(state, args.save_path + f"/epoch_{epoch + 1}.pth")
            logger.info("Epoch: %d, Step: %d, Saving Model to \'%s\'." % (epoch + 1, step, args.save_path))
            print("Epoch: %d, Step: %d, Saving Model to \'%s\'." % (epoch + 1, step, args.save_path))
    
    logger.info("Training finished.")
    print("Training finished.")

    return ppl


def evaluate(model, valid_dataset, stage, criterion):
    model.eval()
    valid_loss = 0.0
    ppl_loss = 0.0
    with torch.no_grad():
        epoch_iterator = tqdm(valid_dataset, ncols=100, leave=False)
        for i, batch in enumerate(epoch_iterator):
            if stage == 1:
                idxs = torch.cat([torch.where(batch['rating']<2)[0], torch.where(batch['rating']>4)[0]])
            elif stage == 2:
                idxs = torch.cat([torch.where(batch['rating']<3)[0], torch.where(batch['rating']>3)[0]])
            else:
                idxs = torch.arange(len(batch['rating']))
            batch = {k: v[idxs].to(device) for k, v in batch.items()}
            ratings = batch['rating'].to(device)
            # import pdb; pdb.set_trace()
            _loss, outputs = model.forward(batch) # [batch_size, seq_len*_max_sent_length*2, vocab_size]
            ppl_loss += torch.mean(_loss.mean(dim=-1)).item()
            outputs = outputs.contiguous() # .view(-1, outputs.shape[-1])
            targets = batch['targets'].contiguous() # .view(-1) # [batch_size, _max_seq_length]
            loss = criterion(outputs, targets, ratings, stage)            
            # loss = criterion(_loss, ratings, stage)
            loss = loss.mean()
            valid_loss += loss.item()
    valid_loss /= len(valid_dataset)
    ppl_loss /= len(valid_dataset)
    ppl = math.exp(ppl_loss)

    return valid_loss, ppl_loss, ppl


def optuna_optimize(trial_times):
    study = optuna.create_study(direction="minimize", study_name="test", storage="sqlite:///optuna.db")
    # >>> optuna-dashboard sqlite:///optuna.db
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

    # optuna_optimize(4)

    time_end = time.time()
    print("Finished!\nTotal time: %s" % format_time(time_end - time_begin))
    











