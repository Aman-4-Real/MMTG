'''
Author: Aman
Date: 2022-03-21 19:38:24
Contact: cq335955781@gmail.com
LastEditors: Aman
LastEditTime: 2022-03-24 13:11:18
'''


import pdb
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



class MultiModalEncoder(nn.Module):
    def __init__(self, model_cfgs):
        super(MultiModalEncoder, self).__init__()
        self.dropout_rate = model_cfgs['dropout']
        self.topic_input_dim = model_cfgs['topic']['input_dim']
        self.topic_hidden_dim = model_cfgs['topic']['hidden_dim']
        self.image_input_dim = model_cfgs['image']['input_dim']
        self.image_hidden_dim = model_cfgs['image']['hidden_dim']
        self.image_num_layers = model_cfgs['image']['num_layers']
        self.text_input_dim = model_cfgs['text']['input_dim']
        self.text_hidden_dim = model_cfgs['text']['hidden_dim']
        self.text_num_layers = model_cfgs['text']['num_layers']
        assert self.topic_hidden_dim == self.image_hidden_dim == self.text_hidden_dim, \
            "The hidden dim of topic, image and text must be equal."
        # for topic mlp
        self.topic_fc = nn.Linear(self.topic_input_dim, self.topic_hidden_dim)
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
        # for text multi-layer rnns
        if model_cfgs['text']['type'] == 'RNN':
            self.rnns_text = nn.RNN(self.text_input_dim, self.text_hidden_dim, \
                                    num_layers=self.text_num_layers, nonlinearity = "relu", dropout=self.dropout_rate)
        elif model_cfgs['text']['type'] == 'LSTM':
            self.rnns_text = nn.LSTM(self.text_input_dim, self.text_hidden_dim, \
                                    num_layers=self.text_num_layers, dropout=self.dropout_rate)
        elif model_cfgs['text']['type'] == 'GRU':
            self.rnns_text = nn.GRU(self.text_input_dim, self.text_hidden_dim, \
                                    num_layers=self.text_num_layers, dropout=self.dropout_rate)
        
        # self.dropout = nn.Dropout(self.dropout_rate)
        self.init_weights()

    def forward(self, encoder_batch):
        '''
        Args:
            encoder_batch: {'topic': [seq_len, batch_size, topic_input_dim], 
                            'image': [seq_len, batch_size, input_dim]
                            'text': [seq_len, batch_size, input_dim]}
        '''
        self.rnns_image.flatten_parameters()
        self.rnns_text.flatten_parameters()
        # Inputs
        x_topic = encoder_batch['topic']
        x_image = encoder_batch['image']
        x_text = encoder_batch['text']

        # Outputs: [seq_len, batch_size, hidden_dim], hidden = [num_layers, batch_size, hidden_dim]
        output_topic = self.topic_fc(x_topic).unsqueeze(0) # [batch_size, topic_input_dim] -> [1, batch_size, topic_hidden_dim]
        output_image, hidden_image = self.rnns_image(x_image) # [seq_len, batch_size, image_input_dim] -> [seq_len, batch_size, image_hidden_dim]
        output_text, hidden_text = self.rnns_text(x_text) # [seq_len, batch_size, text_input_dim] -> [seq_len, batch_size, text_hidden_dim]

        return output_topic, output_image, output_text

    def init_weights(self):
        init.xavier_normal_(self.topic_fc.weight)
        init.xavier_normal_(self.rnns_image.weight_ih_l0)
        init.orthogonal_(self.rnns_image.weight_hh_l0)
        init.xavier_normal_(self.rnns_text.weight_ih_l0)
        init.orthogonal_(self.rnns_text.weight_hh_l0)


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

        # define normal distribution
        self.normal_dists = []
        for i in range(5):
            normal_values = stats.norm.pdf(np.arange(0,5,1), i, 1)
            normal_values = torch.tensor([item/sum(normal_values) for item in normal_values], dtype=torch.float32)
            self.normal_dists.append(normal_values)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

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
        kldivloss = torch.zeros(input.size(1)).to(attention_probs.device)
        for i in range(input.size(1)): # seq_len
            kldivloss[i] = self.kl_loss(attention_probs[:,:,i,:].log(), \
                self.normal_dists[i].to(attention_probs.device).unsqueeze(0).unsqueeze(0).repeat(attention_probs.size(0), attention_probs.size(1), 1))
        
        # print('attention_probs:', attention_probs)
        # [bs, attention_heads, seq_len, seq_len] * [bs, attention_heads, seq_len, attention_head_size] = [bs, attention_heads, seq_len, attention_head_size]
        context_layer = torch.matmul(attention_probs, value_layer) # [bs, attention_heads, seq_len, attention_head_size]
        
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() # [bs, seq_len, attention_heads, attention_head_size]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,) # [bs, seq_len, out_hidden_size]
        context_layer = context_layer.contiguous().view(*new_context_layer_shape)
        
        return context_layer, kldivloss.mean() # [bs, seq_len, out_hidden_size]


class MultiModalAttentionLayer(nn.Module):
    def __init__(self, model_cfgs):
        '''
        Computing the weighted sum of each time step of image and text modality.
        '''
        super(MultiModalAttentionLayer, self).__init__()
        self.seq_len = model_cfgs['seq_len']
        self.topic_hidden_dim = model_cfgs['topic']['hidden_dim']
        self.attention_dim = model_cfgs['MM_ATT']['attention_dim']
        self.att_input_dim = self.topic_hidden_dim

        self.att_matrices = nn.ModuleList([
            nn.Linear(self.att_input_dim, self.attention_dim) for i in range(self.seq_len)
            ])
        # self._norm_fact = 1 / math.sqrt(self.attention_dim)
        self.out_linear = nn.Linear(self.att_input_dim, 2048)

    def forward(self, topic_output, image_output, text_output):
        '''
        Args:
            topic_output: [1, batch_size, hidden_dim]
            image_output, text_output: [seq_len, batch_size, hidden_dim]
        '''
        batch_size = image_output.size(1)
        device = image_output.device
        # Attention
        atten_outputs = torch.zeros(self.seq_len, batch_size, 2048).to(device)
        for i in range(self.seq_len):
            topic_att = self.att_matrices[i](topic_output).transpose(0, 1) # [1, batch_size, hidden_dim] => [batch_size, 1, attention_dim]
            image_att = self.att_matrices[i](image_output[i,:,:].unsqueeze(0)).transpose(0, 1) # [1, batch_size, hidden_dim] => [batch_size, 1, attention_dim]
            text_att = self.att_matrices[i](text_output[i,:,:].unsqueeze(0)).transpose(0, 1) # [1, batch_size, hidden_dim] => [batch_size, 1, attention_dim]
            atten = nn.Softmax(dim=-1)(torch.cat([topic_att, image_att, text_att], dim=1).permute(0,2,1)) # [batch_size, attention_dim, 3] => [batch_size, 1, 3]
            output = torch.bmm(
                atten, torch.cat([topic_output.transpose(0, 1), image_output[i,:,:].unsqueeze(0).transpose(0, 1), \
                        text_output[i,:,:].unsqueeze(0).transpose(0, 1)], dim=1)) # [batch_size, 1, 3] * [batch_size, 3, hidden_dim] => [batch_size, 1, hidden_dim]
            # print(atten)
            atten_out = self.out_linear(output) # [batch_size, 1, hidden_dim] => [batch_size, 1, attention_dim]
            atten_outputs[i,:,:] = atten_out.transpose(0, 1) # [batch_size, 1, attention_dim] => [1, batch_size, attention_dim]
        
        return atten_outputs # [seq_len, batch_size, attention_dim]



class GPT2_Decoder(nn.Module):
    def __init__(
        self,
        model_name="uer/gpt2-chinese-cluecorpussmall",
        config_path="config/model_config.json"
    ):
        super(GPT2_Decoder, self).__init__()
        self.config = GPT2Config.from_json_file(config_path)
        self.token_id2emb = self.load_token_id2emb("vocab/token_id2emb_dict.pkl")
        self.projector_layer1 = nn.Linear(2048, 512)
        self.tanh = nn.Tanh()
        self.projector_layer2 = nn.Linear(512, 768)
        self.gpt2 = GPT2LMHeadModel.from_pretrained(model_name)
        # self.init_weights()

    def load_token_id2emb(self, path):
        token_id2emb = pickle.load(open(path, "rb"))
        return token_id2emb

    # def init_weights(self):
    #     init.xavier_normal_(self.projector_layer1.weight)
    #     init.xavier_normal_(self.projector_layer2.weight)

    def forward(self, concat_output, input_ids, \
                topic_ids, tpw_att_mask, tpw_type_ids, \
                attention_mask=None, type_ids=None, is_train=False):
        '''
        Params:
            concat_output: [batch_size, seq_len, hidden_dim + attention_dim]
            input_ids: [batch_size, seq_len * _sent_length * 2]
            topic_ids: [batch_size, topic_prompt_length]
            tpw_att_mask: [batch_size, topic_prompt_length]
            tpw_type_ids: [batch_size, topic_prompt_length]
            attention_mask: [batch_size, seq_len * _sent_length * 2]
            type_ids: [batch_size, seq_len * _sent_length * 2]
        '''
        # import pdb; pdb.set_trace()
        # process labels
        prompt_length = topic_ids.size(1)
        batch_size = concat_output.size(0)
        seq_len = concat_output.size(1)
        sent_length = 44 # input_ids.size(1)//seq_len
        # labels = torch.cat([torch.zeros(prompt_length, dtype=torch.long).unsqueeze(0).repeat(batch_size,1).to(input_ids.device), input_ids], dim=1)
        labels = torch.cat([topic_ids, input_ids], dim=1)

        # process topic ids
        topic_ids_np = topic_ids.cpu().tolist() # [batch_size, topic_prompt_length]
        topic_ids_wenlan = torch.zeros(batch_size, prompt_length, 2048, dtype=torch.float32).to(topic_ids.device)
        for i in range(batch_size):
            for j in range(prompt_length):
                topic_ids_wenlan[i][j] = torch.tensor(self.token_id2emb[topic_ids_np[i][j]], dtype=torch.float32)

        # process input ids
        input_ids_np = input_ids.cpu().tolist() # [batch_size, _max_sent_length*2]
        input_ids_wenlan = torch.zeros(batch_size, input_ids.size(1), 2048, dtype=torch.float32).to(input_ids.device)
        for i in range(batch_size):
            for j in range(input_ids.size(1)):
                _id = input_ids_np[i][j]
                # if j == input_ids.size(1) - 1:
                input_ids_wenlan[i][j] = torch.tensor(self.token_id2emb[_id], dtype=torch.float32).to(input_ids.device)
                    # continue
                # input_ids_wenlan[i][j] = torch.tensor(self.token_id2emb[_id], dtype=torch.float32).to(input_ids.device) + concat_output[i][j//sent_length]
            for k in range(seq_len):
                input_ids_wenlan[i,sent_length*k:sent_length*(k+1)] = input_ids_wenlan[i,sent_length*k:sent_length*(k+1)] + concat_output[i,k]

        if is_train:
            # process final input embs
            input_embs = torch.cat([topic_ids_wenlan, input_ids_wenlan], dim=1)

            # process type_ids: [1]*44 + [2]*44 + [3]*44 + [4]*44 + [1]*44
            type_ids = torch.cat([tpw_type_ids, type_ids], dim=1).to(input_ids.device) # [seq_len + seq_len * _sent_length * 2]
            
            # process attention mask
            attention_mask = torch.cat([tpw_att_mask, attention_mask], dim=1) # [seq_len + seq_len * _sent_length * 2]
        
            out1 = self.projector_layer1(input_embs)
            out1 = self.tanh(out1)
            gpt_input_embs = self.projector_layer2(out1)
            res = self.gpt2(
                inputs_embeds=gpt_input_embs,
                token_type_ids=type_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            ) # ['loss', 'logits', 'past_key_values']
        
        # inference
        else:
            # import pdb; pdb.set_trace()

            # process final input embs
            input_embs = torch.cat([topic_ids_wenlan, input_ids_wenlan], dim=1)

            _type_ids = tpw_type_ids
            _type_ids_list = list(range(1,concat_output.size(1)))+[1]
            
            # input_embs = torch.cat([input_embs, concat_output[:,0,:].unsqueeze(1)], dim=1)
            # _attention_mask = torch.cat([_attention_mask, torch.ones(1, dtype=torch.long).unsqueeze(0).repeat(concat_output.size(0),1).to(attention_mask.device)], dim=1)
            # input_embs = torch.cat([input_embs, input_ids_wenlan[:,:44,:]], dim=1)
            
            # add type_ids
            for i in range(input_ids.size(1)):
                if (i + 1) % 22 == 0 or (i + 1) % 22 == 1:
                    _type_ids = torch.cat([_type_ids, torch.zeros(1, dtype=torch.long).unsqueeze(0).repeat(type_ids.size(0),1).to(type_ids.device)], dim=1)
                else:
                    cat_type_id = torch.zeros(1, dtype=torch.long) if input_ids[0][i] == 0 else torch.tensor(_type_ids_list[i//sent_length], dtype=torch.long)
                    _type_ids = torch.cat([_type_ids, cat_type_id.unsqueeze(0).repeat(type_ids.size(0),1).to(type_ids.device)], dim=1)
                    
            # add attention mask
            _attention_mask = tpw_att_mask
            for i in range(input_ids.size(1)):
                cat_att_mask = torch.zeros(1, dtype=torch.long) if input_ids[0][i] == 0 else torch.ones(1, dtype=torch.long)
                _attention_mask = torch.cat([_attention_mask, cat_att_mask.unsqueeze(0).repeat(concat_output.size(0),1).to(attention_mask.device)], dim=1)
            
            _labels = torch.zeros(input_embs.size(1), dtype=torch.long).unsqueeze(0).repeat(concat_output.size(0),1).to(input_ids.device)

            out1 = self.projector_layer1(input_embs)
            out1 = self.tanh(out1)
            gpt_input_embs = self.projector_layer2(out1)

            # import pdb; pdb.set_trace()
            # print(input_ids.size(1), gpt_input_embs.shape, _type_ids.shape)
            res = self.gpt2(
                inputs_embeds=gpt_input_embs,
                # token_type_ids=torch.tensor(_type_ids[:,:gpt_input_embs.size(1)], dtype=torch.long).unsqueeze(0),
                token_type_ids=torch.tensor(_type_ids, dtype=torch.long),
                attention_mask=torch.tensor(_attention_mask, dtype=torch.long),
                labels=_labels,
                return_dict=True
            )
        # print("loss: ", res['loss'])
        return res


class EXPTeller(nn.Module):
    def __init__(self, model_cfgs, vocab_size, train_flag=False):
        super(EXPTeller, self).__init__()
        self.model_cfgs = model_cfgs
        self.vocab_size = vocab_size
        self.encoder = MultiModalEncoder(model_cfgs)
        self.ln_layer1 = torch.nn.LayerNorm(model_cfgs['topic']['hidden_dim'], elementwise_affine=True)
        self.ln_layer2 = torch.nn.LayerNorm(model_cfgs['image']['hidden_dim'], elementwise_affine=True)
        self.ln_layer3 = torch.nn.LayerNorm(model_cfgs['text']['hidden_dim'], elementwise_affine=True)
        self.img_inner_atten_layer = InnerModalAttentionLayer(model_cfgs)
        self.text_inner_atten_layer = InnerModalAttentionLayer(model_cfgs)
        self.mm_atten_layer = MultiModalAttentionLayer(model_cfgs)
        # self.agg_layer = AggregateLayer(model_cfgs)
        self.decoder = GPT2_Decoder()
        self.train_flag = train_flag
        if train_flag:
            # Load pre-trained GPT2 model
            print("Loading pre-trained GPT2 model...")
            state_dict = torch.load(model_cfgs['GPT2_PATH'], map_location="cpu")
            if 'state_dict' in state_dict:
                state_dict = {
                    key: value for key, value in state_dict["state_dict"].items()
                }
            self.decoder.load_state_dict(state_dict)
            print("Pre-trained GPT2 model loaded.")
            
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
        encoder_batch = {'topic': batch['topic_emb'].float(), \
                         'image': batch['img_embs'].transpose(0, 1).float(), \
                         'text': batch['r_embs'].transpose(0, 1).float()}
        batch_size = batch['img_embs'].size(0)
        device = batch['img_embs'].device
        seq_len = batch['img_embs'].size(1)
        # output_len = batch['targets'].size(2)
        
        # ===== Multi-modal Encoder =====
        topic_output, image_output, text_output = self.encoder(encoder_batch)
        # topic_output: [1, batch_size, hidden_dim], image_output, text_output: [seq_len, batch_size, hidden_dim]
        # print("topic_output: ", topic_output)
        
        topic_output = self.ln_layer1(topic_output)
        image_output = self.ln_layer2(image_output)
        text_output = self.ln_layer3(text_output)
        
        # ===== Self Attention Layer =====
        # print("img att: ")
        img_inner_attention_output, img_kl_loss = self.img_inner_atten_layer(image_output.transpose(0, 1))
        # print("text att: ")
        text_inner_attention_output, text_kl_loss = self.text_inner_atten_layer(text_output.transpose(0, 1))
        # [batch_size, seq_len, attention_dim] => [batch_size, seq_len, attention_dim]
        
        # print("img_kl_loss:", img_kl_loss, "text_kl_loss:", text_kl_loss)
        # print(img_self_attention_output)


        # ===== Multi-modal Attention Layer =====
        mm_attention_output = self.mm_atten_layer(topic_output, \
            img_inner_attention_output.transpose(0,1), text_inner_attention_output.transpose(0,1))
        # [seq_len, batch_size, attention_dim]
        
        # ===== Aggregate =====
        # output_agg = self.agg_layer(mm_attention_output).transpose(0, 1)
        # [seq_len, batch_size, attention_dim] => [batch_size, seq_len, attention_dim]

        # ===== Concatenate topic_output and output_agg =====
        # topic_output_repeat = topic_output.repeat(seq_len, 1, 1).transpose(0, 1) # [batch_size, seq_len, hidden_dim]
        # concat_output = torch.cat([topic_output_repeat, output_agg], dim=-1) # [batch_size, seq_len, hidden_dim + attention_dim]
        # print(concat_output)
        
        # ===== Decoder =====
        decoder_input = batch['targets'] # [batch_size, seq_len * _sent_length * 2]
        # outputs = torch.zeros(batch_size, seq_len, decoder_input.shape[2], self.vocab_size).to(device)
        # loss = torch.zeros(batch_size, seq_len).to(device)

        res = self.decoder(mm_attention_output.transpose(0, 1), decoder_input, \
                        batch['topic_ids'], batch['tpw_attention_mask'], batch['tpw_type_ids'], \
                        batch['attention_mask'], batch['type_ids'], self.train_flag)
        loss, outputs = res['loss'], res['logits']

        return loss, (img_kl_loss + text_kl_loss).mean(), outputs # [batch_size, seq_len+_max_seq_length, vocab_size]


