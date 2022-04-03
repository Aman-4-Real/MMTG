'''
Author: Aman
Date: 2022-04-03 21:43:38
Contact: cq335955781@gmail.com
LastEditors: Aman
LastEditTime: 2022-04-03 23:07:01
'''


# type: [str], RNN type
# input_dim: [int], input dimension
# hidden_dim: [int], hidden dimension
# seq_dim: [int], number of cells per layer
# num_layers: [int], number of layers
model_cfgs = {
    'seq_len': 5,
    'image': {
        'type': 'GRU',
        'input_dim': 2048,
        'hidden_dim': 256,
        'num_layers': 1
    },
    'SELF_ATT': {
        'hidden_size': 256, # topic, image, text and self att hidden dims must be equal
        'attention_heads': 4
    },
    'Decoder': {
        'embedding_dim': 512,
    },
    'dropout': 0.1
}

class data_config():
    def __init__(self):
        self.max_sent_length = 20
        self.max_seq_length = 220

