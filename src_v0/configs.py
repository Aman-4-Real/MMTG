'''
Author: Aman
Date: 2021-11-15 11:13:35
Contact: cq335955781@gmail.com
LastEditors: Aman
LastEditTime: 2021-11-20 22:16:15
'''


# type: [str], RNN type
# input_dim: [int], input dimension
# hidden_dim: [int], hidden dimension
# seq_dim: [int], number of cells per layer
# num_layers: [int], number of layers
model_cfgs = {
    'topic': {
        'type': 'LSTM',
        'input_dim': 2048,
        'hidden_dim': 512,
        'num_layers': 1
    },
    'image': {
        'type': 'LSTM',
        'input_dim': 2048,
        'hidden_dim': 512,
        'num_layers': 4
    },
    'text': {
        'type': 'LSTM',
        'input_dim': 2048,
        'hidden_dim': 512,
        'num_layers': 4
    },
    'agg': {
        'type': 'GRU',
        'output_dim': 512,
        'num_layers': 1
    },
    'decoder': {
        'type': 'GRU',
        'hidden_dim': 512,
        'num_layers': 1,
        'embedding_dim': 256
    }
}

class data_config():
    def __init__(self):
        self.max_sent_length = 20
        self.max_seq_length = 150

# class cfgs():
#     def __init__(self):
#         self.num_labels_1 = 3
#         self.num_labels_2 = 10
#         self.linear_hidden_size = 256
#         self.linear_dropout = 0.3



        

















