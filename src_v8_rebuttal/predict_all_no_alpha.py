'''
Author: Aman
Date: 2022-06-08 19:36:55
Contact: cq335955781@gmail.com
LastEditors: Aman
LastEditTime: 2022-06-10 13:51:08
'''


import argparse
import math
import os
import pdb
import time as t

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import BertTokenizer

from configs import data_config, model_cfgs
from model_no_alpha import EXPTeller
from MyDataset import MyDataset
from utils import *



def _is_word(word):
    for item in list(word):
        if item not in "qwertyuiopasdfghjklzxcvbnm":
            return False
    return True


def _is_chinese_char(char):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    cp = ord(char)
    if (
        (cp >= 0x4E00 and cp <= 0x9FFF)
        or (cp >= 0x3400 and cp <= 0x4DBF)  #
        or (cp >= 0x20000 and cp <= 0x2A6DF)  #
        or (cp >= 0x2A700 and cp <= 0x2B73F)  #
        or (cp >= 0x2B740 and cp <= 0x2B81F)  #
        or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
        or (cp >= 0xF900 and cp <= 0xFAFF)
        or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
    ):  #
        return True

    return False




def beam_decode(
    model,
    inputs,
    length,
    tokenizer,
    beam_size=5,
    temperature=1.0,
    repitition_penalty=1.0,
    device='cpu'):
    
    start_input = inputs['targets'] # [1, seq_len, 1]
    start_len = start_input.shape[2]

    seq_beams = [[(start_input[0,i,:].cpu().tolist(), 0)] for i in range(seq_len)]
    with torch.no_grad():
        for l_step in trange(start_len, length):
            next_beams = [[] for _ in range(seq_len)]
            beams = len(seq_beams[0])
            for beam_i in range(beams):
                tmp_targets = torch.zeros(seq_len, len(seq_beams[0][beam_i][0]), dtype=torch.long).unsqueeze(0).to(device)
                # import pdb; pdb.set_trace()
                for i in range(seq_len):
                    tmp_targets[0][i] = torch.tensor(seq_beams[i][beam_i][0], dtype=torch.long, device=device)
                inputs['targets'] = tmp_targets
                _, outputs = model.forward(inputs) # [batch_size, seq_len, now_sent_length, vocab_size]
                next_token_logits = outputs[0, :, -1, :8102] # [seq_len, vocab_size]
                # next_tokens = torch.zeros([seq_len, 1], dtype=torch.long).to(device)
                generated = inputs['targets']
                for i in range(seq_len):
                    for id in set(generated[0][i]): # penalty for repetition
                        next_token_logits[i][id] /= repitition_penalty
                    next_token_logits[i] = next_token_logits[i] / temperature
                    # print(min(next_token_logits[i]))
                    next_token_logits[i][tokenizer.convert_tokens_to_ids("[UNK]")] = -50 # -float("Inf")
                    token_probs = F.softmax(next_token_logits[i], dim=-1)
                    new_sequences = []
                    # append new tokens to old sequences and re-score
                    old_seq = seq_beams[i][beam_i][0]
                    old_score = seq_beams[i][beam_i][1]
                    for char_id in range(len(token_probs)):
                        new_seq = old_seq + [char_id]
                        # considering log-likelihood for scoring
                        new_score = old_score + math.log(token_probs[char_id].item())
                        new_sequences.append((new_seq, new_score))
                    # next_beams[i] += sorted(new_sequences, key = lambda val: val[1], reverse = True)[:beam_size]
                    next_beams[i] += new_sequences
            for i in range(seq_len):
                seq_beams[i] = sorted(next_beams[i], key = lambda val: val[1], reverse = True)[:beam_size]
    # return the best sequence
    res = [seq_beams[i][0][0] for i in range(seq_len)]

    return res



def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (vocabulary size)
        top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert (
        logits.dim() == 1
    )  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(
    model,
    start_input,
    length,
    tokenizer,
    temperature=1.0,
    beam_size=5,
    top_k=30,
    top_p=0.0,
    repitition_penalty=1.0,
    device="cpu"
):
    inputs = start_input
    for k, v in inputs.items():
        if k == 'targets':
            inputs[k] = torch.tensor(v, dtype=torch.long, device=device).unsqueeze(0)
        else:
            inputs[k] = torch.tensor(v, dtype=torch.float32, device=device).unsqueeze(0)
    generated = inputs['targets']
    if beam_size > 0:
        generated = beam_decode(model, inputs, length, tokenizer, beam_size, \
                                temperature=1.0, repitition_penalty=1.0, device=device)
    else:
        with torch.no_grad():
            for i in range(length):
                if i > 0 and (i + 2) % 22 == 0: # add [#EOS#]
                    inputs['targets'] = torch.cat((inputs['targets'], torch.tensor([[2]]).to(inputs['targets'].device)), dim=-1)
                    continue
                if i > 0 and (i + 2) % 22 == 1: # add [#START#]
                    inputs['targets'] = torch.cat((inputs['targets'], torch.tensor([[1]]).to(inputs['targets'].device)), dim=-1)
                    continue
                _, outputs = model.forward(inputs) # [batch_size, seq_len+_max_seq_length, vocab_size]
                next_token_logits = outputs[0, -1, :] # [batch_size, vocab_size]
                generated = inputs['targets']
                for id in set(generated[0]):
                    # import pdb; pdb.set_trace()
                    if id in [0, 102]: # skip punctuation
                        continue
                    next_token_logits[id] /= repitition_penalty
                next_token_logits = next_token_logits / temperature
                next_token_logits[tokenizer.convert_tokens_to_ids("[#START#]")] = -float("Inf")
                next_token_logits[tokenizer.convert_tokens_to_ids("[#EOS#]")] = -float("Inf")
                next_token_logits[tokenizer.convert_tokens_to_ids("[UNK]")] = -float("Inf")
                next_token_logits[tokenizer.convert_tokens_to_ids("[SEP]")] = -float("Inf")
                if generated[0][-1] == 0:
                    next_token = torch.tensor([0]).to(generated.device).unsqueeze(0)
                else:
                    filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)[:13317]
                    next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1).unsqueeze(0)
                inputs['targets'] = torch.cat((generated, next_token), dim=-1)
                # import pdb; pdb.set_trace()
            generated = generated.tolist()[0]
    return generated



def calculate_sent_prob(model, start_input, label, length, device):
    # import pdb; pdb.set_trace()
    inputs = start_input
    for k, v in inputs.items():
        if k == 'targets':
            if len(v) == 1:
                inputs[k] = torch.tensor(v, dtype=torch.long, device=device).unsqueeze(0)
            else:
                inputs[k] = torch.tensor(v, dtype=torch.long, device=device)
        elif k == 'img_embs' or k == 'r_embs':
            if len(v) == 2:
                inputs[k] = v.clone().detach().unsqueeze(0)
            # else:
            #     inputs[k] = v.clone().detach()
        else:
            if len(v) == 1:
                inputs[k] = v.clone().detach().unsqueeze(0)
            # else:
            #     inputs[k] = v.clone().detach()
    generated = inputs['targets']
    probs = [[] for _ in range(len(generated))]
    max_probs = [[] for _ in range(len(generated))]
    with torch.no_grad():
        for t in range(length-1):
            # import pdb; pdb.set_trace()
            _, _, outputs = model.forward(inputs) # [batch_size, now_sent_length, vocab_size]
            curr_token = torch.zeros(outputs.size(0), dtype=torch.long, device=device).unsqueeze(1)
            for i in range(outputs.size(0)):
                next_token_logits = outputs[i, -1, :] # [vocab_size]
                # generated = inputs['targets'][i]
                probabilities = F.softmax(next_token_logits, dim=-1)
                probs[i].append(-np.log(probabilities[label[i][t+1]].item()))
                max_probs[i].append(-np.log(np.max(probabilities.cpu().numpy())))
                next_token = torch.tensor(label[i][t+1], dtype=torch.long, device=device).unsqueeze(0)
                # import pdb; pdb.set_trace()
                curr_token[i] = next_token
            inputs['targets'] = torch.cat((inputs['targets'], curr_token), dim=-1)
            # import pdb; pdb.set_trace()
        # generated = generated.tolist()[0]
    return probs, max_probs



def test(model, test_dataset, device):
    model.eval()
    ppl_loss = 0.0
    with torch.no_grad():
        epoch_iterator = tqdm(test_dataset, ncols=100, leave=False)
        for i, batch in enumerate(epoch_iterator):
            batch = {k: v.to(device) for k, v in batch.items()}
            _loss, _, outputs = model.forward(batch) # [batch_size, seq_len*_max_sent_length*2, vocab_size]
            ppl_loss += torch.mean(_loss.mean(dim=-1)).item()
            outputs = outputs.contiguous()
            target = batch['targets'].contiguous() # [batch_size*seq_len*_max_sent_length*2]
    ppl_loss /= len(test_dataset)
    ppl = math.exp(ppl_loss)

    return ppl_loss, ppl



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_ids", default="0,1,2,3", type=str, help="GPU device ids")
    parser.add_argument("--CUDA_VISIBLE_DEVICES", default="2,1,0,3", type=str, help="CUDA_VISIBLE_DEVICES")
    parser.add_argument("--batch_size", default=32, type=int, help="Test batch size")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--num_workers", default=8, type=int, help="Number of workers")
    parser.add_argument("--data_path", default="../datasets/new_data_rating/final_test_50.pkl", type=str, help="Data directory")
    parser.add_argument("--model_path", default="/data/caoqian/gen_exp/src_v8_rebuttal/models/lr1e-5_bs96_kl02_add_no_alpha/epoch_5.pth", type=str, help="Model path")
    parser.add_argument("--tokenizer_path", default="./vocab/vocab.txt", type=str, required=False, help="词表路径")
    parser.add_argument("--beam_size", default=0, type=int, required=False, help="beam search size") # 20: 13min
    parser.add_argument("--temperature", default=1.1, type=float, required=False, help="生成温度")
    parser.add_argument("--topk", default=10, type=int, required=False, help="最高几选一")
    parser.add_argument("--topp", default=0.7, type=float, required=False, help="最高积累概率")
    parser.add_argument("--repetition_penalty", default=1.5, type=float, required=False)
    parser.add_argument("--n_samples", default=10, type=int, required=False, help="生成的样本数量")
    # parser.add_argument("--save_samples", action="store_true", help="保存产生的样本")
    parser.add_argument("--save_samples_path", default="res/lr1e-5_bs96_kl02_add_tk10_tp07_tm1o1_rpt1o5_no_alpha.txt", type=str, required=False, help="保存样本的路径")
    

    # global args
    global model_cfgs, data_config
    model_cfgs = model_cfgs
    data_config = data_config()
    
    args = parser.parse_args()
    # print("args:\n" + args.__repr__())
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_VISIBLE_DEVICES # args.device_ids
    device_ids = [int(item) for item in args.device_ids.split(",")]
    beam_size = args.beam_size
    batch_size = args.batch_size
    num_workers = args.num_workers
    temperature = args.temperature
    topk = args.topk
    topp = args.topp
    repetition_penalty = args.repetition_penalty
    length = data_config.max_seq_length # 200
    n_samples = args.n_samples

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path)
    # print("vocab_size: ", len(tokenizer.vocab))
    
    # load model
    checkpoint = torch.load(args.model_path)
    model = EXPTeller(model_cfgs, len(tokenizer.vocab), False) # predicting mode
    model.to(device)
    model = nn.DataParallel(model, device_ids=device_ids)
    model.load_state_dict(checkpoint['model'])
    # print("Loaded model from {}".format(args.model_path))

    # print("Loading data...")
    test_data_file = args.data_path
    test_data = MyDataset(test_data_file, tokenizer, data_config, False)
    test_dataset = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    # print("Data test loaded.")


    while 0:
        ppl_loss, ppl = test(model, test_dataset, device)
        print("PPL loss: %.4f, PPL: %.4f" % (ppl_loss, ppl))
        break
    
    
    # =====> generate samples <=====
    while 1:
        f1 = open(args.save_samples_path, "w", encoding="utf-8")
        # f2 = open("res/labels_cl_ln_lr1e-5_ep3.txt", "w", encoding="utf-8")
        for idx in trange(0,len(test_dataset.dataset),1): # len(test_dataset.dataset)
            n_preds = []
            for _ in range(n_samples):
                encoded = [tokenizer.convert_tokens_to_ids('[#START#]')] # Input [#START#] token
                start_input = test_dataset.dataset[idx]
                start_input['targets'] = np.asarray(encoded)
                preds = sample_sequence(
                    model,
                    start_input,
                    length=length,
                    tokenizer=tokenizer,
                    temperature=temperature,
                    beam_size=beam_size,
                    top_k=topk,
                    top_p=topp,
                    repitition_penalty=repetition_penalty,
                    device=device,
                )
                preds = [tokenizer.convert_ids_to_tokens(line) for line in preds]
                all_idx_of_eos = [i for i,v in enumerate(preds) if v=='[#EOS#]']
                if len(all_idx_of_eos) >= 10 and '[SEP]' not in preds[:all_idx_of_eos[-1]]:
                    eos_idx = all_idx_of_eos[9]
                    preds = preds[:eos_idx+1] + ['[SEP]']
                elif '[SEP]' in preds:
                    sep_idx = preds.index('[SEP]')
                    preds = preds[:sep_idx+1]
                else:
                    preds = preds + ['[SEP]']
                # print(("".join(preds[:-2])+'[SEP]').replace('[#EOS#]', '，').replace('[#START#]', '').replace('[SEP]', ''))
                # tmp = ''.join(preds).replace('[#EOS#]', '，').replace('[#START#]', '').replace('[SEP]', '')
                tmp = ''.join(preds).replace('[SEP]', '').replace('[PAD]', '').replace('[#START#]', '').replace('[#EOS#]', '，')
                while tmp[-1] == '，':
                    tmp = tmp[:-1]
                n_preds += [tmp]
                
            label = test_dataset.dataset[idx]['targets']
            label_tokens = tokenizer.convert_ids_to_tokens(label)
            sep_idx = label_tokens.index('[SEP]')
            label_tokens = label_tokens[:sep_idx+1]
            # label_out = ("".join(label_tokens[:-2])+'[SEP]').replace('[#EOS#]', '，').replace('[#START#]', '').replace('[SEP]', '')
            # print(n_preds)
            # print(label_out)
            for j in range(len(n_preds)):
                f1.write(n_preds[j]+'\n')
                # f2.write(label_out+'\n')
        f1.close()
        # f2.close()
        break
        


    # =====> Calculate the PPL of label and top-1 prediction of the test data <=====
    while 0:
        label_probs = []
        top1_probs = []
        epoch_iterator = tqdm(test_dataset, ncols=100, leave=False)
        for batch in epoch_iterator:
            labels = batch['targets'].tolist() # bs * 200
            label_tokens = [tokenizer.convert_ids_to_tokens(label) for label in labels]
            encoded = [[tokenizer.convert_tokens_to_ids('[#START#]')] for i in range(batch['topic_emb'].size(0))] # Input [#START#] token
            start_input = batch
            start_input['targets'] = np.asarray(encoded)
            probs, maxprobs = calculate_sent_prob(model, start_input, labels, length, device)
            valid_len = [labels[j].index(102) + 1 for j in range(batch['topic_emb'].size(0))]
            label_probs += [sum(probs[j][:valid_len[j]])/valid_len[j] for j in range(batch['topic_emb'].size(0))]
            top1_probs += [sum(maxprobs[j][:valid_len[j]])/valid_len[j] for j in range(batch['topic_emb'].size(0))]
        print("Label Probs:", np.mean(label_probs), "Top-1 Probs:", np.mean(top1_probs))
        print("Label PPL:", np.exp(np.mean(label_probs)), "Top-1 PPL:", np.exp(np.mean(top1_probs)))
        break



if __name__ == "__main__":
    main()

