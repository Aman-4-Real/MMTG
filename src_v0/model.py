'''
Author: Aman
Date: 2021-11-15 10:40:56
Contact: cq335955781@gmail.com
LastEditors: Aman
LastEditTime: 2021-12-01 17:53:57
'''

import torch
import torch.nn as nn
import torch.nn.init as init
import random
import numpy as np

class MyRNNsEncoder(nn.Module):
    def __init__(self, model_cfgs, dropout=0.3):
        super(MyRNNsEncoder, self).__init__()
        self.model_cfgs = model_cfgs
        self.dropout = dropout
        self.topic_input_dim = model_cfgs['topic']['input_dim']
        self.topic_hidden_dim = model_cfgs['topic']['hidden_dim']
        self.topic_num_layers = model_cfgs['topic']['num_layers']
        self.image_input_dim = model_cfgs['image']['input_dim']
        self.image_hidden_dim = model_cfgs['image']['hidden_dim']
        self.image_num_layers = model_cfgs['image']['num_layers']
        self.text_input_dim = model_cfgs['text']['input_dim']
        self.text_hidden_dim = model_cfgs['text']['hidden_dim']
        self.text_num_layers = model_cfgs['text']['num_layers']
        self.agg_output_dim = model_cfgs['agg']['output_dim']
        self.agg_num_layers = model_cfgs['agg']['num_layers']
        # for topic multi-layer rnns
        if self.model_cfgs['topic']['type'] == 'RNN':
            self.rnns_topic = nn.RNN(self.topic_input_dim, self.topic_hidden_dim, \
                                    num_layers=self.topic_num_layers, nonlinearity = "relu", dropout=self.dropout)
        elif self.model_cfgs['topic']['type'] == 'LSTM':
            self.rnns_topic = nn.LSTM(self.topic_input_dim, self.topic_hidden_dim, \
                                    num_layers=self.topic_num_layers, dropout=self.dropout)
        elif self.model_cfgs['topic']['type'] == 'GRU':
            self.rnns_topic = nn.GRU(self.topic_input_dim, self.topic_hidden_dim, \
                                    num_layers=self.topic_num_layers, dropout=self.dropout)
        # for image multi-layer rnns
        if self.model_cfgs['image']['type'] == 'RNN':
            self.rnns_image = nn.RNN(self.image_input_dim, self.image_hidden_dim, \
                                    num_layers=self.image_num_layers, nonlinearity = "relu", dropout=self.dropout)
        elif self.model_cfgs['image']['type'] == 'LSTM':
            self.rnns_image = nn.LSTM(self.image_input_dim, self.image_hidden_dim, \
                                    num_layers=self.image_num_layers, dropout=self.dropout)
        elif self.model_cfgs['image']['type'] == 'GRU':
            self.rnns_image = nn.GRU(self.image_input_dim, self.image_hidden_dim, \
                                    num_layers=self.image_num_layers, dropout=self.dropout)
        # for text multi-layer rnns
        if self.model_cfgs['text']['type'] == 'RNN':
            self.rnns_text = nn.RNN(self.text_input_dim, self.text_hidden_dim, \
                                    num_layers=self.text_num_layers, nonlinearity = "relu", dropout=self.dropout)
        elif self.model_cfgs['text']['type'] == 'LSTM':
            self.rnns_text = nn.LSTM(self.text_input_dim, self.text_hidden_dim, \
                                    num_layers=self.text_num_layers, dropout=self.dropout)
        elif self.model_cfgs['text']['type'] == 'GRU':
            self.rnns_text = nn.GRU(self.text_input_dim, self.text_hidden_dim, \
                                    num_layers=self.text_num_layers, dropout=self.dropout)
        # for aggregate multi-layer rnns
        if self.model_cfgs['agg']['type'] == 'RNN':
            self.agg_layer = nn.RNN(self.topic_hidden_dim+self.image_hidden_dim+self.text_hidden_dim, \
                                    self.agg_output_dim, num_layers=self.agg_num_layers, nonlinearity = "relu", dropout=self.dropout)
        elif self.model_cfgs['agg']['type'] == 'LSTM':
            self.agg_layer = nn.LSTM(self.topic_hidden_dim+self.image_hidden_dim+self.text_hidden_dim, \
                                     self.agg_output_dim, num_layers=self.agg_num_layers, dropout=self.dropout)
        elif self.model_cfgs['agg']['type'] == 'GRU':
            self.agg_layer = nn.GRU(self.topic_hidden_dim+self.image_hidden_dim+self.text_hidden_dim, \
                                    self.agg_output_dim, num_layers=self.agg_num_layers, dropout=self.dropout)

        self.dropout = nn.Dropout(self.dropout)
        self.init_weights()

    def forward(self, encoder_batch):
        '''
        Args:
            batch: {'topic': [seq_len, batch_size, input_dim], ...}
        '''
        # Inputs
        x_topic = encoder_batch['topic']
        x_image = encoder_batch['image']
        x_text = encoder_batch['text']

        self.rnns_topic.flatten_parameters()
        self.rnns_image.flatten_parameters()
        self.rnns_text.flatten_parameters()
        self.agg_layer.flatten_parameters()

        # Outputs
        # outputs = [seq_len, batch_size, hidden_dim * n_directions]
        # hidden = [num_layers * n_directions, batch_size, hidden_dim]
        output_topic, hidden_topic = self.rnns_topic(x_topic)
        output_image, hidden_image = self.rnns_image(x_image)
        output_text, hidden_text = self.rnns_text(x_text)
        # Concatenate
        outputs = torch.cat((output_topic, output_image, output_text), dim=2)
        # Aggregate
        output_agg, hidden_agg = self.agg_layer(outputs)
        # Dropout
        # output_agg = self.dropout(output_agg)

        return hidden_agg # [1, batch_size, hidden_dim]

    def init_weights(self):
        init.xavier_normal_(self.rnns_topic.weight_ih_l0)
        init.orthogonal_(self.rnns_topic.weight_hh_l0)
        init.xavier_normal_(self.rnns_image.weight_ih_l0)
        init.orthogonal_(self.rnns_image.weight_hh_l0)
        init.xavier_normal_(self.rnns_text.weight_ih_l0)
        init.orthogonal_(self.rnns_text.weight_hh_l0)



class MyRNNsDecoder(nn.Module):
    def __init__(self, model_cfgs, vocab_size, dropout=0.2):
        super(MyRNNsDecoder, self).__init__()
        self.model_cfgs = model_cfgs
        self.dropout = dropout
        self.agg_output_dim = model_cfgs['agg']['output_dim']
        self.embedding_dim = model_cfgs['decoder']['embedding_dim']
        self.decoder_hidden_dim = model_cfgs['decoder']['hidden_dim']
        self.decoder_num_layers = model_cfgs['decoder']['num_layers']
        self.decoder_output_dim = vocab_size
        assert self.decoder_hidden_dim == self.agg_output_dim, \
            "The hidden dim of decoder must be equal to the hidden dim of agg layer"
        # embedding layer
        self.embeddings = nn.Embedding(vocab_size, self.embedding_dim)
        # for decoder multi-layer rnns
        if self.model_cfgs['decoder']['type'] == 'RNN':
            self.decoder = nn.RNN(self.agg_output_dim + self.embedding_dim, self.decoder_hidden_dim, \
                                  num_layers=self.decoder_num_layers, nonlinearity = "relu", dropout=self.dropout)
        elif self.model_cfgs['decoder']['type'] == 'LSTM':
            self.decoder = nn.LSTM(self.agg_output_dim + self.embedding_dim, self.decoder_hidden_dim, \
                                   num_layers=self.decoder_num_layers, dropout=self.dropout)
        elif self.model_cfgs['decoder']['type'] == 'GRU':
            self.decoder = nn.GRU(self.agg_output_dim + self.embedding_dim, self.decoder_hidden_dim, \
                                  num_layers=self.decoder_num_layers, dropout=self.dropout)
        else:
            raise ValueError('Decoder RNN type not supported')

        self.out = nn.Linear(self.embedding_dim + self.agg_output_dim + self.decoder_hidden_dim, self.decoder_output_dim)
        self.dropout = nn.Dropout(self.dropout)
        self.init_weights()

    def forward(self, input, hidden, enc_outputs_agg):
        '''
        Args:
            input: [batch_size]
            hidden: [n_layers * n_directions, batch_size, hiddem_dim]
            enc_outputs_agg (context): [n_layers * n_directions, batch_size, agg_output_dim]
        '''
        self.decoder.flatten_parameters()
        # Embedding
        # input = input.unsqueeze(0)
        embedded = self.dropout(self.embeddings(input)).unsqueeze(0) # [1, batch_size, embedding_dim]
        # Concatenate
        embedded_agg = torch.cat((embedded, enc_outputs_agg), dim = 2) # [1, batch_size, embedding_dim + agg_output_dim]
        # Outputs
        # outputs: [seq_len, batch_size, hidden_dim * n_directions]
        # hidden: [num_layers * n_directions, batch_size, hidden_dim]
        output, hidden = self.decoder(embedded_agg, hidden)
        # Dropout
        output = self.dropout(output)
        output = torch.cat((embedded.squeeze(0), hidden.squeeze(0), enc_outputs_agg.squeeze(0)), dim = 1)
        # Linear
        logits = self.out(output)

        return logits, hidden # logits: [batch_size, vocab_size], hidden: [1, batch_size, hidden_dim]
    
    def init_weights(self):
        init.xavier_normal_(self.decoder.weight_ih_l0)
        init.orthogonal_(self.decoder.weight_hh_l0)
        init.xavier_normal_(self.out.weight)
        init.constant_(self.out.bias, 0)



class MySeq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(MySeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, batch, teacher_forcing_ratio=0.5):
        '''
        Args:
            batch: {
                'topic_emb': [seq_len, batch_size, input_dim],
                'img_embs': [seq_len, batch_size, input_dim],
                'r_embs': [seq_len, batch_size, input_dim],
                'target': [batch_size, input_dim],
            }
        '''
        seq_len = batch['img_embs'].size(1)
        batch_size, output_len = batch['target'].size()
        device = batch['target'].device
        encoder_batch = {'topic': batch['topic_emb'].unsqueeze(0).repeat(seq_len,1,1).float(), \
                         'image': batch['img_embs'].transpose(0,1).float(), \
                         'text': batch['r_embs'].transpose(0,1).float()}
        # Encoder
        enc_outputs_agg = self.encoder(encoder_batch) # [1, batch_size, agg_output_dim]
        # Decoder
        decoder_hidden = enc_outputs_agg
        decoder_input = batch['target'].transpose(0,1)[0, :] # [batch_size]
        decoder_outputs = torch.zeros(output_len, batch_size, self.decoder.decoder_output_dim).to(device)
        for i in range(1, output_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, enc_outputs_agg)
            # decoder_output: [batch_size, vocab_size], hidden: [1, batch_size, hidden_dim]
            teacher_force = random.random() < teacher_forcing_ratio
            topv, topi = decoder_output.topk(1) # values, indices
            decoder_input = (batch['target'][:, i] if teacher_force else topi.view(-1))
            decoder_outputs[i] = decoder_output

        return decoder_outputs.transpose(0, 1) # [batch_size, output_len, vocab_size]




# Codes in RNN-Cells
# self.input_dim = input_dim
# self.hidden_dims = hidden_dims
# self.seq_dim = seq_dim
# self.layer_dim = layer_dim
# self.output_dim = output_dim
# self.dropout = dropout
# self._input_dim = [input_dim] + hidden_dims + [output_dim]
# self.layer_cells_lists = nn.ModuleList() # each sublist is a horizontal layer
# if self.type == 'RNN':
#     for i in range(layer_dim):
#         cell = nn.RNNCell(self._input_dim[i], self._input_dim[i+1], nonlinearity = "relu")
#         layer_cells = nn.ModuleList([cell] * seq_dim)
#         self.layer_cells_lists.append(layer_cells)
# elif self.type == 'LSTM':
#     for i in range(layer_dim):
#         cell = nn.LSTMCell(self._input_dim[i], self._input_dim[i+1])
#         layer_cells = nn.ModuleList([cell] * seq_dim)
#         self.layer_cells_lists.append(layer_cells)
# elif self.type == 'GRU':
#     for i in range(layer_dim):
#         cell = nn.GRUCell(self._input_dim[i], self._input_dim[i+1])
#         layer_cells = nn.ModuleList([cell] * seq_dim)
#         self.layer_cells_lists.append(layer_cells)
# else:
#     raise ValueError('RNN type not supported')