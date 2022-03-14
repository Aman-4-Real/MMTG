'''
Author: Aman
Date: 2021-11-15 11:14:05
Contact: cq335955781@gmail.com
LastEditors: Aman
LastEditTime: 2021-11-24 19:53:50
'''

from torch.utils.data import Dataset
import numpy as np
import pickle

class MyDataset(Dataset):
    def __init__(self, file_path, tokenizer, data_config):
        super(MyDataset, self).__init__()
        self._filename = file_path
        f = open(file_path, 'rb')
        self.data = pickle.load(f)
        f.close()
        self._tokenizer = tokenizer
        self._max_sent_length = data_config.max_sent_length
        self._max_seq_length = data_config.max_seq_length
        self._total_len = len(self.data)
    
    def __len__(self):
        return self._total_len

    def __getitem__(self, idx):
        '''
        item.keys:
            'topic', 'topic_emb', 'lyrics', 
            'text_0', 'text_0_emb', 'text_1', 'text_1_emb', 'text_2', 'text_2_emb', 'text_3', 'text_3_emb', 'text_4', 'text_4_emb',
            'img_0', 'img_0_emb', 'img_1', 'img_1_emb', 'img_2', 'img_2_emb', 'img_3', 'img_3_emb', 'img_4', 'img_4_emb',
            'r_0', 'r_0_emb', 'r_1', 'r_1_emb', 'r_2', 'r_2_emb', 'r_3', 'r_3_emb', 'r_4', 'r_4_emb'
        '''
        topic_emb = self.data[idx]['topic_emb']
        img_embs = [self.data[idx]['img_' + str(i) + '_emb'] for i in range(5)]
        r_embs = [self.data[idx]['r_' + str(i) + '_emb'] for i in range(5)]
        target = self.convert_lyrics2ids(self.data[idx]['lyrics'])
        batch = {
            'topic_emb': np.asarray(topic_emb),
            'img_embs': np.asarray(img_embs),
            'r_embs': np.asarray(r_embs),
            'target': np.asarray(target)
        }
        return batch

    def convert_lyrics2ids(self, lyrics):
        '''
        lyrics: list of str
        '''
        all_tokens = ['[#START#]']
        for sent in lyrics:
            sent = sent.replace(' ', '')
            sent = sent.replace('\n', '')
            sent = sent.replace('\t', '')
            sent = sent.replace('\r', '')
            sent = sent.replace('\xa0', '')
            sent = sent.replace('\u3000', '')
            sent = self._tokenizer.tokenize(sent)[:self._max_sent_length]
            all_tokens.extend(sent)
            all_tokens.append('[#EOS#]')
        all_tokens.append(self._tokenizer.sep_token)
        # print('\n',all_tokens)
        all_tokens = all_tokens[:self._max_seq_length]
        while len(all_tokens) < self._max_seq_length:
            all_tokens.append(self._tokenizer.pad_token)
        all_tokens = self._tokenizer.convert_tokens_to_ids(all_tokens)

        return all_tokens
