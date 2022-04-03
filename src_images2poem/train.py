'''
Author: Aman
Date: 2022-04-03 21:43:38
Contact: cq335955781@gmail.com
LastEditors: Aman
LastEditTime: 2022-04-03 23:21:37
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
from model import images2poem
from MyDataset import MyDataset
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,0,1"


parser = argparse.ArgumentParser()
parser.add_argument("--device_ids", default="[0,1,2,3]", type=str, help="GPU device ids")
parser.add_argument("--batch_size", default=256, type=int, help="Batch size")
parser.add_argument("--val_batch_size", default=256, type=int, help="Eval batch size")
parser.add_argument("--epochs", default=5, type=int, help="Number of epochs")
parser.add_argument("--lr", default=1e-05, type=float, help="Learning rate")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--num_workers", default=8, type=int, help="Number of workers")
parser.add_argument("--log_interval", default=100, type=int, help="Log interval")
parser.add_argument("--val_interval_ratio", default=0.2, type=float, help="Eval once every interval ratio of training data")
parser.add_argument("--train_data_path", default="../datasets/new_data_rating/train_data_with_ratings_210k.pkl", type=str, help="Train data path")
parser.add_argument("--val_data_path", default="../datasets/new_data_rating/val_data_with_ratings_8k.pkl", type=str, help="Val data path")
parser.add_argument("--save_model", default=True, type=bool, help="Save model or not")
parser.add_argument("--save_path", default="./models/lr1e-5_bs256", type=str, help="Save directory")
# parser.add_argument("--save_interval", default=1, type=int, help="Save interval")
parser.add_argument("--log_path", default="./logs/lr1e-5_bs256.log", type=str, help="Log directory")
parser.add_argument("--tensorboard_log_dir", default="./logs/lr1e-5_bs256", type=str, help="Tensorboard log directory")

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
        args.lr = trial.suggest_discrete_uniform("lr", 5e-6, 2.5e-5, 5e-6)
        args.batch_size = trial.suggest_categorical("batch_size", [32, 64, 96])
        # args.dropout = trial.suggest_uniform("dropout", 0.0, 0.5)
        # args.epochs = trial.suggest_int("epochs", 5, 15)
        # model_cfgs['image']['num_layers'] = trial.suggest_int("image_rnn_num_layers", 2, 8)


    model = images2poem(model_cfgs, len(tokenizer.vocab), True)
    
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



# class MyLoss(torch.nn.Module):
#     def __init__(self):
#         super(MyLoss, self).__init__()
#         self._max_topic_len = data_config.topic_prompt_length
#         self._seq_len = model_cfgs['seq_len']

#     def forward(self, outputs, targets, ratings, stage):
#         '''
#         Args:
#             outputs: (batch_size, topic_prompt_length + seq_len + _max_seq_length, vocab_size)
#             targets: (batch_size, _max_seq_length)
#             ratings: (batch_size)
#         '''
#         # import pdb; pdb.set_trace()
#         NEAR_0 = 1e-10
#         device = outputs.device
#         zero = torch.zeros_like(ratings)
#         one = torch.ones_like(ratings)
#         batch_size = targets.shape[0]
#         if stage == 1:
#             ratings = torch.where(ratings > 4, one, zero)
#         else:
#             ratings = torch.where(ratings > 3, one, zero)

#         shift_logits = outputs[:, self._max_topic_len:-1, :]
#         shift_labels = targets[:, 1:]
        
#         # Flatten the tokens
#         loss_fct = nn.CrossEntropyLoss()
        
#         loss = torch.zeros(batch_size).to(device)
#         for i in range(batch_size):
#             y = ratings[i]
#             _loss = loss_fct(shift_logits[i], shift_labels[i])
#             p = 1/torch.exp(_loss)
#             # import pdb; pdb.set_trace()
#             loss[i] += torch.sum(- y * torch.log(p + NEAR_0) - (1 - y) * torch.log(1 - p + NEAR_0))
#         # import pdb; pdb.set_trace()
#         return torch.mean(loss)
         


def train(model, train_data, valid_data):
    # args.tensorboard_log_dir = f"logs/optuna_10trials/5ep_lr{args.lr}".replace('.','e') + f"_bsz{args.batch_size}"
    # if not os.path.exists(args.tensorboard_log_dir):
    #     os.makedirs(args.tensorboard_log_dir)
    # writer = SummaryWriter(args.tensorboard_log_dir)
    # args.save_path = f"models/optuna_10trials/5ep_lr{args.lr}".replace('.','e') + f"_bsz{args.batch_size}" # "./models/optuna_10trials" + 

    print("Now lr is ", args.lr, "Now batch_size is ", args.batch_size)
    logger.info('Now lr is %s, batch_size is %s.' % (args.lr, args.batch_size))
    
    train_dataset = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    valid_dataset = DataLoader(valid_data, batch_size=val_batch_size, shuffle=True, num_workers=args.num_workers)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    training_steps = len(train_dataset) * args.epochs
    print('Total training steps:', training_steps)
    logger.info('* number of training steps: %d' % training_steps) # number of training steps
    one_epoch_steps = len(train_dataset)
    # warmup and decay the learning rate
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = int(one_epoch_steps * 0.1), 
                                                num_training_steps = training_steps)
    # criterion = nn.BCEWithLogitsLoss()
    # criterion = nn.CrossEntropyLoss()
    # criterion = MyLoss()
    best_val_loss = float("inf")
    global_steps = 0
    for epoch in range(args.epochs):
        t1 = time.time()
        torch.cuda.empty_cache()
        print("\nEpoch ", epoch + 1, "/", args.epochs)
        logger.info("Epoch " + str(epoch + 1) + "/" + str(args.epochs))
        epoch_iterator = tqdm(enumerate(train_dataset),
                                desc="%s: %d/%d Epochs >> Steps" % ("Train", epoch + 1, args.epochs),
                                total=len(train_dataset),
                                bar_format="{l_bar}{r_bar}")
        avg_loss = 0.0
        model.train()
        # Setting the tqdm progress bar
        for step, batch in epoch_iterator:
            # import pdb; pdb.set_trace()
            idxs = torch.where(batch['rating']>4)[0]
            batch = {k: v[idxs].to(device) for k, v in batch.items()}
            ratings = batch['rating'].to(device)
            loss, outputs = model.forward(batch) # [batch_size, seq_len+_max_seq_length, vocab_size]
            # import pdb; pdb.set_trace()
            # outputs = outputs.contiguous() # .view(-1, outputs.shape[-1])
            targets = batch['targets'].contiguous() # .view(-1) # [batch_size, _max_seq_length]
            # ppl_loss = torch.mean(_loss.mean(dim=-1))
            # loss = criterion(outputs, targets, ratings, stage)
            # loss = criterion(outputs, targets)
            total_loss = loss.mean()
            # print("---loss: ", total_loss.item())
            # import pdb; pdb.set_trace()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # clip gradient
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            for param_group in optimizer.param_groups:
                args.lr = param_group['lr']
            epoch_iterator.set_postfix(lr=args.lr, ppl=math.exp(total_loss), loss=total_loss.item())  # show the learning rate and loss on the progress bar
            global_steps += 1
            writer.add_scalar('train/loss', total_loss.item(), global_steps)
            # writer.add_scalar('train/ppl_loss', ppl_loss, global_steps)
            writer.add_scalar('train/ppl', math.exp(total_loss.item()), global_steps)
            writer.add_scalar('train/lr', args.lr, global_steps)
            if step > 0 and (step + 1) % (len(train_dataset) * args.val_interval_ratio) == 0:
                val_loss, ppl = evaluate(model, valid_dataset)
                logger.info("Epoch: %d, Step: %d/%d, Val. Loss: %.4f, Val. PPL: %.3f" % (epoch + 1, step + 1, len(train_dataset), val_loss, ppl))
                print(" Epoch: %d, Step: %d/%d, Val. Loss: %.4f, Val. PPL: %.3f" % (epoch + 1, step + 1, len(train_dataset), val_loss, ppl))
                writer.add_scalar('eval/loss', val_loss, global_steps)
                # writer.add_scalar('eval/ppl_loss', ppl_loss, global_steps)
                writer.add_scalar('eval/ppl', ppl, global_steps)
                # writer.add_scalar('eval/kldiv_loss', kldiv_loss, global_steps)
                # Save model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if args.save_model:
                        if not os.path.exists(args.save_path):
                            os.makedirs(args.save_path)
                        state = {'model': model.state_dict(), 'args': args, 'model_cfgs': model_cfgs} # 'optimizer': optimizer.state_dict(), 
                        torch.save(state, args.save_path + f"/best_val_model.pth") # loss_{best_val_loss:.3f}
                        logger.info("Epoch: %d, Step: %d, Saving Model to \'%s\'." % (epoch + 1, step, args.save_path))
                        print("Epoch: %d, Step: %d, Saving Model to \'%s\'." % (epoch + 1, step, args.save_path))
                model.train()
            avg_loss += total_loss.item()
            if step > 0 and (step + 1) % args.log_interval == 0:
                logger.info("Epoch: %d, Step: %d/%d, Average loss: %.6f" % (epoch + 1, step + 1, len(train_dataset), avg_loss / (step + 1)))
        # End of epoch
        val_loss, ppl = evaluate(model, valid_dataset)
        logger.info("End eval of epoch %d. Val. Loss: %.4f, Val. PPL: %.2f" % (epoch + 1, val_loss, ppl))
        print("End eval of epoch %d. Val. Loss: %.4f, Val. PPL: %.2f" % (epoch + 1, val_loss, ppl))
        writer.add_scalar('eval/loss', val_loss, global_steps)
        # writer.add_scalar('eval/ppl_loss', ppl_loss, global_steps)
        writer.add_scalar('eval/ppl', ppl, global_steps)
        # writer.add_scalar('eval/kldiv_loss', kldiv_loss, global_steps)
        model.train()
        logger.info("Average loss: %.4f  Elapsed time: %s" % (avg_loss / (len(train_dataset) + 1), format_time(time.time()-t1)))
        print("Average loss: %.4f  Elapsed time: %s" % (avg_loss / (len(train_dataset) + 1), format_time(time.time()-t1)))
        if args.save_model:
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            state = {'model': model.state_dict(), 'args': args, 'model_cfgs': model_cfgs} # 'optimizer': optimizer.state_dict(), 
            torch.save(state, args.save_path + f"/epoch_{epoch + 1}.pth")
            logger.info("Epoch: %d, Step: %d, Saving Model to \'%s\'." % (epoch + 1, step, args.save_path))
            print("Epoch: %d, Step: %d, Saving Model to \'%s\'." % (epoch + 1, step, args.save_path))
    
    logger.info("Training finished.")
    print("Training finished.")

    return val_loss


def evaluate(model, valid_dataset):
    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        epoch_iterator = tqdm(valid_dataset, ncols=100, leave=False)
        for i, batch in enumerate(epoch_iterator):
            idxs = torch.where(batch['rating']>4)[0]
            batch = {k: v[idxs].to(device) for k, v in batch.items()}
            ratings = batch['rating'].to(device)
            # import pdb; pdb.set_trace()
            loss, outputs = model.forward(batch) # [batch_size, seq_len*_max_sent_length*2, vocab_size]
            # outputs = outputs.contiguous() # .view(-1, outputs.shape[-1])
            targets = batch['targets'].contiguous() # .view(-1) # [batch_size, _max_seq_length]
            # loss = criterion(outputs, targets, ratings, stage)          
            # loss = criterion(_loss, ratings, stage)
            valid_loss += loss.mean().item()
    valid_loss /= len(valid_dataset)
    ppl = math.exp(valid_loss)

    return valid_loss, ppl


def optuna_optimize(trial_times):
    study = optuna.create_study(direction="minimize", study_name="10trials", storage="sqlite:///optuna.db")
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

    # optuna_optimize(10)

    time_end = time.time()
    print("Finished!\nTotal time: %s" % format_time(time_end - time_begin))
    











