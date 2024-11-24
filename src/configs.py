'''
Author: Aman
Date: 2022-03-21 19:38:24
Contact: cq335955781@gmail.com
LastEditors: Aman
LastEditTime: 2022-08-10 19:18:19
'''


# type: [str], RNN type, RNN, LSTM, GRU, or TRM
# input_dim: [int], input dimension
# hidden_dim: [int], hidden dimension
# num_layers: [int], number of layers
model_cfgs = {
    'seq_len': 5, # 10 lyrics sentences = seq_len * 2
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
    'GPT2_PATH': './pretrained/GPT2_lyrics_ckpt_epoch00.ckpt',
    'dropout': 0.1
}

class data_config():
    def __init__(self):
        self.topic_prompt_length = 15
        self.max_sent_length = 20
        self.max_seq_length = 220
        self.wenlan_emb_size = 2048

    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except Exception as e:
            print("No {} exists!".format(key))
