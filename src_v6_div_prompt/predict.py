'''
Author: Aman
Date: 2022-03-15 13:50:46
Contact: cq335955781@gmail.com
LastEditors: Aman
LastEditTime: 2022-03-16 13:31:44
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
from model import EXPTeller
from MyDataset import MyDataset
from utils import *

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"



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
                _, _, outputs = model.forward(inputs) # [batch_size, seq_len, now_sent_length, vocab_size]
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
            for _ in range(length-inputs['targets'].size(1)):
                # print(inputs['targets'].shape)
                if inputs['targets'].size(1) % 45 == 0:
                    inputs['targets'] = torch.cat([inputs['targets'], \
                        torch.tensor([1], dtype=torch.long, device=device).unsqueeze(0)], dim=1)
                    # import pdb; pdb.set_trace()
                _, _, outputs = model.forward(inputs) # [batch_size, seq_len+_max_seq_length, vocab_size]
                
                # outputs = model.forward(mm_attention_output, inputs['targets'], \
                #             inputs['topic_ids'], inputs['tpw_attention_mask'], inputs['tpw_type_ids'], \
                #             False)['logits']
                # import pdb; pdb.set_trace()
                next_token_logits = outputs[0, -1, :] # [batch_size, vocab_size]
                generated = inputs['targets']
                for id in set(generated[0]):
                    # import pdb; pdb.set_trace()
                    if id in [0, 1, 2, 102]: # skip punctuation
                        continue
                    next_token_logits[id] /= repitition_penalty
                next_token_logits = next_token_logits / temperature
                # import pdb; pdb.set_trace()
                next_token_logits[tokenizer.convert_tokens_to_ids("[UNK]")] = -float("Inf")
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
            inputs[k] = torch.tensor(v, dtype=torch.long, device=device).unsqueeze(0)
        else:
            inputs[k] = torch.tensor(v, dtype=torch.float32, device=device).unsqueeze(0)
    generated = inputs['targets']
    probs = []
    max_probs = []
    with torch.no_grad():
        for t in range(length-1):
            # import pdb; pdb.set_trace()
            _, _, outputs = model.forward(inputs) # [batch_size, seq_len, now_sent_length, vocab_size]
            next_token_logits = outputs[0, -1, :] # [seq_len, vocab_size]
            # next_tokens = torch.zeros([1], dtype=torch.long).to(device)
            generated = inputs['targets']
            
            probabilities = F.softmax(next_token_logits, dim=-1)
            probs.append(-np.log(probabilities[label[t+1]].item()))
            max_probs.append(-np.log(np.max(probabilities.cpu().numpy())))
            next_tokens = torch.tensor(label[t+1], dtype=torch.long, device=device).unsqueeze(0).unsqueeze(0)
            # import pdb; pdb.set_trace()
            inputs['targets'] = torch.cat((generated, next_tokens), dim=-1)
            # import pdb; pdb.set_trace()
        # generated = generated.tolist()[0]
    return probs, max_probs



def test(model, test_dataset, device):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    _loss = 0.0
    with torch.no_grad():
        epoch_iterator = tqdm(test_dataset, ncols=100, leave=False)
        for i, batch in enumerate(epoch_iterator):
            batch = {k: v.to(device) for k, v in batch.items()}
            _, _, outputs = model.forward(batch) # [batch_size, seq_len, _max_sent_length*2, vocab_size]
            outputs = outputs.contiguous().view(-1, outputs.shape[-1])
            target = batch['targets'].contiguous().view(-1) # [batch_size*seq_len*_max_sent_length*2]
            loss = criterion(outputs, target)
            loss = loss.mean()
            _loss += loss.item()
    _loss /= len(test_dataset)
    ppl = math.exp(_loss)

    return _loss, ppl



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_ids", default="0,1,2,3", type=str, help="GPU device ids")
    parser.add_argument("--batch_size", default=128, type=int, help="Test batch size")
    parser.add_argument("--seed", default=42, type=int, help="Random seed")
    parser.add_argument("--num_workers", default=4, type=int, help="Number of workers")
    parser.add_argument("--data_path", default="../datasets/new_data_rating/final_test_50.pkl", type=str, help="Data directory")
    parser.add_argument("--model_path", default="./models/lr1e-6_bs96_kl02/best_val_model.pth", type=str, help="Model path")
    parser.add_argument("--tokenizer_path", default="./vocab/vocab.txt", type=str, required=False, help="词表路径")
    parser.add_argument("--beam_size", default=0, type=int, required=False, help="beam search size") # 20: 13min
    parser.add_argument("--temperature", default=1.1, type=float, required=False, help="生成温度")
    parser.add_argument("--topk", default=10, type=int, required=False, help="最高几选一")
    parser.add_argument("--topp", default=0.7, type=float, required=False, help="最高积累概率")
    parser.add_argument("--repetition_penalty", default=1.1, type=float, required=False)
    # parser.add_argument("--prefix", default="今天", type=str, required=False, help="生成文章的开头")
    parser.add_argument("--save_samples", action="store_true", help="保存产生的样本")
    parser.add_argument("--save_samples_path", default=".", type=str, required=False, help="保存样本的路径")
    parser.add_argument("--n_samples", default=5, type=int, required=False, help="生成的样本数量")
    

    # global args
    global model_cfgs, data_config
    model_cfgs = model_cfgs
    model_cfgs['dropout'] = 0
    data_config = data_config()
    
    args = parser.parse_args()
    print("args:\n" + args.__repr__())
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_ids  # 此处设置程序使用哪些显卡
    device_ids = [int(item) for item in args.device_ids.split(",")]
    beam_size = args.beam_size
    batch_size = args.batch_size
    num_workers = args.num_workers
    temperature = args.temperature
    topk = args.topk
    topp = args.topp
    repetition_penalty = args.repetition_penalty
    length = data_config.max_seq_length # 200

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_path)
    print("vocab_size: ", len(tokenizer.vocab))
    # import pdb; pdb.set_trace()
    
    # load model
    checkpoint = torch.load(args.model_path)
    model = EXPTeller(model_cfgs, len(tokenizer.vocab), False) # predicting mode
    model.to(device)
    model = nn.DataParallel(model, device_ids=device_ids)
    model.load_state_dict(checkpoint['model'])
    print("Loaded model from {}".format(args.model_path))


    print("Loading data...")
    test_data_file = args.data_path
    test_data = MyDataset(test_data_file, tokenizer, data_config, False)
    test_dataset = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print("Data test loaded.")
    # test_loss, ppl = test(model, test_dataset, device)
    # print("Test loss: %.4f, PPL: %.4f" % (test_loss, ppl))
    
    def swap(item, m, n):
        item[[m,n], :] = item[[n,m], :]
        # print(item)
        # item[:,:] = 9
        return item


    print("Now displaying any instance of the test data. 0 <= idx < %d" % (len(test_data)))
    while True:
        # input a idx
        idx = eval(input("Input a idx: "))
        print("="*100)
        print("idx:", idx)
        topic_list = ''.join(tokenizer.convert_ids_to_tokens(test_dataset.dataset[idx]['topic_ids']))
        print(topic_list.replace('[PAD]', ''))
        for _ in range(args.n_samples):
            label = test_dataset.dataset[idx]['targets']
            label_tokens = tokenizer.convert_ids_to_tokens(label)
            # encoded = [tokenizer.convert_tokens_to_ids('[#START#]我们')] # Input [#START#] token
            start_input = test_dataset.dataset[idx]
            # start_input['targets'] = np.asarray(encoded)
            start_input['targets'] = np.asarray([])
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
            # all_idx_of_eos = [i for i,v in enumerate(preds) if v=='[#EOS#]']
            # if len(all_idx_of_eos) >= 10 and '[SEP]' not in preds[:all_idx_of_eos[-1]]:
            #     eos_idx = all_idx_of_eos[9]
            #     preds = preds[:eos_idx+1] + ['[SEP]']
            # elif '[SEP]' in preds:
            #     sep_idx = preds.index('[SEP]')
            #     preds = preds[:sep_idx+1]
            # else:
            #     preds = preds + ['[SEP]']
            # print("Prediction:")
            # sep_idx = preds.index('[SEP]')
            # preds = preds[:sep_idx]
            print(" ".join(preds))
            print("-"*80)
        
        while 0:
            try:
                start_input = test_dataset.dataset[idx]
                start_input['targets'] = np.asarray(encoded)
                # print(start_input['img_embs'])
                m, n = input("Swap: ").split()
                swap_input = start_input
                swap_input['img_embs'] = swap(swap_input['img_embs'], int(m), int(n))
                swap_input['r_embs'] = swap(swap_input['r_embs'], int(m), int(n))
                # print(swap_input['img_embs'])
                preds = sample_sequence(
                    model,
                    swap_input,
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
                print("Swap Prediction:")
                sep_idx = preds.index('[SEP]')
                preds = preds[:sep_idx]
                print(" ".join(preds))
            except:
                break
        

        print("Label:")
        sep_idx = label_tokens.index('[SEP]')
        label_tokens = label_tokens[:sep_idx]
        print(" ".join(label_tokens))

        # ===================== for probs caculation =====================
        # print("-"*80)
        # import pdb; pdb.set_trace()
        # start_input = test_dataset.dataset[idx]
        # start_input['targets'] = np.asarray(encoded)
        # probs, maxprobs = calculate_sent_prob(model, start_input, label, length, device)
        # valid_len = test_dataset.dataset[idx]['targets'].tolist().index(102) + 1
        # print("Label Probs:", sum(probs[:valid_len])/valid_len, \
        #       "Top-1 Probs:", sum(maxprobs[:valid_len])/valid_len)
        
        print("="*100)
    


if __name__ == "__main__":
    main()

