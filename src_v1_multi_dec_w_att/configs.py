'''
Author: Aman
Date: 2021-11-15 11:13:35
Contact: cq335955781@gmail.com
LastEditors: Aman
LastEditTime: 2021-12-03 02:07:11
'''


# type: [str], RNN type
# input_dim: [int], input dimension
# hidden_dim: [int], hidden dimension
# seq_dim: [int], number of cells per layer
# num_layers: [int], number of layers
model_cfgs = {
    'seq_len': 5,
    'topic': {
        'input_dim': 2048,
        'hidden_dim': 256
    },
    'image': {
        'type': 'GRU',
        'input_dim': 2048,
        'hidden_dim': 256,
        'num_layers': 1
    },
    'text': {
        'type': 'GRU',
        'input_dim': 2048,
        'hidden_dim': 256, # topic, image, text hidden dims must be equal
        'num_layers': 1
    },
    'MM_ATT': {
        'attention_dim': 512
    },
    'SELF_ATT': {
        'hidden_size': 512, # equals with MM_ATT.hidden_size
        'attention_heads': 4
    },
    'agg': {
        'type': 'GRU',
        'output_dim': 512,
        'num_layers': 1
    },
    'decoder': {
        'type': 'GRU',
        'hidden_dim': 512, # equals with agg.output_dim
        'num_layers': 1,
        'embedding_dim': 512
    },
    'dropout': 0.2
}

class data_config():
    def __init__(self):
        self.max_sent_length = 20
        # self.max_seq_length = 150




        

















