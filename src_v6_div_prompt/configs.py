'''
Author: Aman
Date: 2022-03-14 20:44:37
Contact: cq335955781@gmail.com
LastEditors: Aman
LastEditTime: 2022-03-14 21:22:31
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
        'hidden_dim': 512
    },
    'image': {
        'type': 'GRU',
        'input_dim': 2048,
        'hidden_dim': 512,
        'num_layers': 1
    },
    'text': {
        'type': 'GRU',
        'input_dim': 2048,
        'hidden_dim': 512, 
        'num_layers': 1
    },
    'SELF_ATT': {
        'hidden_size': 512, # topic, image, text and self att hidden dims must be equal
        'attention_heads': 4
    },
    'MM_ATT': {
        'attention_dim': 1
    },
    # 'agg': {
    #     'type': 'GRU',
    #     'output_dim': 1536,
    #     'num_layers': 1
    # },
    'GPT2_PATH': '/home/caoqian/playground/GPT2-Chinese-master/5ep_tsteps5w_410335_cleaned_data/epoch00.ckpt',
    'dropout': 0.1
}

class data_config():
    def __init__(self):
        self.topic_prompt_length = 15
        self.max_sent_length = 20
        # self.max_seq_length = 200

