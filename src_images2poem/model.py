'''
Author: Aman
Date: 2022-04-03 21:43:38
Contact: cq335955781@gmail.com
LastEditors: Aman
LastEditTime: 2022-04-03 23:12:32
'''

import torch
import torch.nn as nn
import torch.nn.init as init
from scipy import stats
import random
import math
import pickle
import numpy as np

from transformers import GPT2LMHeadModel, GPT2Config
from configs import data_config



class ImageEncoder(nn.Module):
    def __init__(self, model_cfgs):
        super(ImageEncoder, self).__init__()
        self.dropout_rate = model_cfgs['dropout']
        self.image_input_dim = model_cfgs['image']['input_dim']
        self.image_hidden_dim = model_cfgs['image']['hidden_dim']
        self.image_num_layers = model_cfgs['image']['num_layers']
        # for image multi-layer rnns
        if model_cfgs['image']['type'] == 'RNN':
            self.rnns_image = nn.RNN(self.image_input_dim, self.image_hidden_dim, \
                                    num_layers=self.image_num_layers, nonlinearity = "relu", dropout=self.dropout_rate)
        elif model_cfgs['image']['type'] == 'LSTM':
            self.rnns_image = nn.LSTM(self.image_input_dim, self.image_hidden_dim, \
                                    num_layers=self.image_num_layers, dropout=self.dropout_rate)
        elif model_cfgs['image']['type'] == 'GRU':
            self.rnns_image = nn.GRU(self.image_input_dim, self.image_hidden_dim, \
                                    num_layers=self.image_num_layers, dropout=self.dropout_rate)
        # self.dropout = nn.Dropout(self.dropout_rate)
        self.init_weights()

    def forward(self, encoder_batch):
        '''
        Args:
            encoder_batch: {'image': [seq_len, batch_size, input_dim]}
        '''
        self.rnns_image.flatten_parameters()
        # Inputs
        x_image = encoder_batch['image']

        # Outputs: [seq_len, batch_size, hidden_dim], hidden = [num_layers, batch_size, hidden_dim]
        output_image, hidden_image = self.rnns_image(x_image) # [seq_len, batch_size, image_input_dim] -> [seq_len, batch_size, image_hidden_dim]

        return output_image

    def init_weights(self):
        init.xavier_normal_(self.rnns_image.weight_ih_l0)
        init.orthogonal_(self.rnns_image.weight_hh_l0)


class InnerModalAttentionLayer(nn.Module):
    def __init__(self, model_cfgs):
        '''
        Compute the self attention of the hidden states of the image and text inputs.
        '''
        super(InnerModalAttentionLayer, self).__init__()
        self.hidden_size = model_cfgs['SELF_ATT']['hidden_size']
        self.attention_heads = model_cfgs['SELF_ATT']['attention_heads']
        self.dropout_rate = model_cfgs['dropout']

        if self.hidden_size % self.attention_heads != 0:
            raise ValueError("The hidden size (%d) is not a multiple of the number of attention heads (%d)" % (self.hidden_size, self.attention_heads))

        self.attention_heads = self.attention_heads
        self.attention_head_size = self.hidden_size // self.attention_heads
        self.all_head_size = int(self.attention_heads * self.attention_head_size) # all_head_size = hidden_size
        
        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)
        
        # dropout
        # self.dropout = nn.Dropout(self.dropout_rate)

    def reshape_for_scores(self, x):
        '''
        Reshape the weight matrix to multi-heads form.
        Args:
            x: [bs, seq_len, hid_size]
        '''
        new_x_shape = x.size()[:-1] + (self.attention_heads, self.attention_head_size) # [bs, seq_len, attention_heads, attention_head_size]
        x = x.contiguous().view(*new_x_shape)
        # print("x:", x)
        return x.permute(0, 2, 1, 3).contiguous() # [bs, attention_heads, seq_len, attention_head_size]

    def forward(self, input):
        '''
        Args:
            input: [batch_size, seq_len, attention_dim]
        '''
        mixed_query_layer = self.query(input)
        mixed_key_layer = self.key(input)
        mixed_value_layer = self.value(input) # [bs, seq_len, hidden_size]
        
        query_layer = self.reshape_for_scores(mixed_query_layer)
        key_layer = self.reshape_for_scores(mixed_key_layer)
        value_layer = self.reshape_for_scores(mixed_value_layer) # [bs, attention_heads, seq_len, attention_head_size]
        # print("/////")
        # print('value_layer:', value_layer)
        
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # [bs, attention_heads, seq_len, seq_len]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size) # [bs, attention_heads, seq_len, seq_len]
        # print(attention_scores)
        attention_probs = nn.Softmax(dim=-1)(attention_scores) # [bs, attention_heads, seq_len, seq_len]
        # print(attention_probs)
        # attention_probs = self.dropout(attention_probs)
        
        # print('attention_probs:', attention_probs)
        # [bs, attention_heads, seq_len, seq_len] * [bs, attention_heads, seq_len, attention_head_size] = [bs, attention_heads, seq_len, attention_head_size]
        context_layer = torch.matmul(attention_probs, value_layer) # [bs, attention_heads, seq_len, attention_head_size]
        
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() # [bs, seq_len, attention_heads, attention_head_size]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,) # [bs, seq_len, out_hidden_size]
        context_layer = context_layer.contiguous().view(*new_context_layer_shape)
        
        return context_layer # [bs, seq_len, out_hidden_size]


class GPT2_Decoder(nn.Module):
    def __init__(
        self,
        model_cfgs,
        model_name="uer/gpt2-chinese-cluecorpussmall",
        config_path="config/model_config.json"
    ):
        super(GPT2_Decoder, self).__init__()
        self.config = GPT2Config.from_json_file(config_path)
        # self.token_id2emb = self.load_token_id2emb("vocab/token_id2emb_dict.pkl")
        # self.projector_layer1 = nn.Linear(2048, 512)
        # self.tanh = nn.Tanh()
        # self.projector_layer2 = nn.Linear(512, 768)
        self.gpt2 = GPT2LMHeadModel.from_pretrained(model_name)
        # self.init_weights()

        self.embedding_dim = model_cfgs['Decoder']['embedding_dim']
        self.embeddings = nn.Embedding(self.config.vocab_size, self.embedding_dim)

    # def load_token_id2emb(self, path):
    #     token_id2emb = pickle.load(open(path, "rb"))
    #     return token_id2emb

    # def init_weights(self):
    #     init.xavier_normal_(self.projector_layer1.weight)
    #     init.xavier_normal_(self.projector_layer2.weight)

    def forward(self, img_output, input_ids, is_train=False):
        '''
        Params:
            img_output: [batch_size, 1, hidden_dim + attention_dim]
            input_ids: [batch_size, seq_len * _sent_length * 2]
        '''
        # import pdb; pdb.set_trace()
        # process labels
        batch_size = img_output.size(0)
        labels = input_ids

        embedded = self.embeddings(labels) # [batch_size, length, embedding_dim]
        img_embedded = torch.cat([img_output.unsqueeze(1).repeat(1, labels.size(1), 1), embedded], dim=-1) # [batch_size, length, embedding_dim + hidden_dim]

        if is_train:
            res = self.gpt2(
                inputs_embeds=img_embedded,
                labels=labels,
                return_dict=True
            ) # ['loss', 'logits', 'past_key_values']
        # inference
        else:
            res = self.gpt2(
                inputs_embeds=img_embedded,
                labels=labels,
                return_dict=True
            ) # ['loss', 'logits', 'past_key_values']
        # print("loss: ", res['loss'])
        return res


class images2poem(nn.Module):
    def __init__(self, model_cfgs, vocab_size, train_flag=False):
        super(images2poem, self).__init__()
        self.model_cfgs = model_cfgs
        self.vocab_size = vocab_size
        self.encoder = ImageEncoder(model_cfgs)
        self.img_inner_atten_layer = InnerModalAttentionLayer(model_cfgs)
        # self.agg_layer = AggregateLayer(model_cfgs)
        self.decoder = GPT2_Decoder(model_cfgs)
        self.train_flag = train_flag
        # if train_flag:
        #     # Load pre-trained GPT2 model
        #     print("Loading pre-trained GPT2 model...")
        #     state_dict = torch.load(model_cfgs['GPT2_PATH'], map_location="cpu")
        #     if 'state_dict' in state_dict:
        #         state_dict = {
        #             key: value for key, value in state_dict["state_dict"].items()
        #         }
        #     self.decoder.load_state_dict(state_dict)
        #     print("Pre-trained GPT2 model loaded.")
            
    def forward(self, batch):
        '''
        Args:
            batch: {
                'topic_ids': [batch_size, topic_prompt_length],
                'tpw_attention_mask': [batch_size, topic_prompt_length],
                'tpw_type_ids': [batch_size, topic_prompt_length],
                'topic_emb': [batch_size, input_dim],
                'img_embs': [batch_size, seq_len, input_dim],
                'r_embs': [batch_size, seq_len, input_dim],
                'targets': [batch_size, seq_len * _max_sent_length * 2],
                'attention_mask': [batch_size, seq_len * _max_sent_length * 2],
                'type_ids': [batch_size, seq_len * _max_sent_length * 2],
            }
        '''
        encoder_batch = {'image': batch['img_embs'].transpose(0, 1).float()}
        batch_size = batch['img_embs'].size(0)
        device = batch['img_embs'].device
        seq_len = batch['img_embs'].size(1)
        # output_len = batch['targets'].size(2)
        
        # ===== Multi-modal Encoder =====
        image_output = self.encoder(encoder_batch)
        # image_output: [seq_len, batch_size, hidden_dim]
        
        # ===== Self Attention Layer =====
        img_inner_attention_output = self.img_inner_atten_layer(image_output.transpose(0, 1))
        # [batch_size, seq_len, attention_dim] => [batch_size, seq_len, attention_dim]
        avg_img_embbed = img_inner_attention_output.mean(dim=1)
        
        # ===== Decoder =====
        decoder_input = batch['targets'] # [batch_size, seq_len * _sent_length * 2]
        # outputs = torch.zeros(batch_size, seq_len, decoder_input.shape[2], self.vocab_size).to(device)
        # loss = torch.zeros(batch_size, seq_len).to(device)

        res = self.decoder(avg_img_embbed, decoder_input, self.train_flag)
        loss, outputs = res['loss'], res['logits']

        return loss, outputs # [batch_size, seq_len+_max_seq_length, vocab_size]


