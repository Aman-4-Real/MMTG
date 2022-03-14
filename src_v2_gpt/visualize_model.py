'''
Author: Aman
Date: 2021-11-29 10:14:08
Contact: cq335955781@gmail.com
LastEditors: Aman
LastEditTime: 2021-12-14 10:09:52
'''

import torch
import hiddenlayer as h
from configs import model_cfgs, data_config
from model import EXPTeller
from transformers import BertTokenizer
import numpy as np
from MyDataset import MyDataset
from torch.utils.data import DataLoader

# load tokenizer
tokenizer = BertTokenizer.from_pretrained("./vocab/vocab.txt")
data_config = data_config()

test_data_file = "../datasets/sample_data/data_test_283.pkl"
test_data = MyDataset(test_data_file, tokenizer, data_config)
test_dataset = DataLoader(test_data, batch_size=2, shuffle=False, num_workers=2)
for batch in test_dataset:
    break
# batch = {
#     'topic_emb': torch.randn(4, 2048),
#     'img_embs': torch.randn(4, 5, 2048),
#     'r_embs': torch.randn(4, 5, 2048),
#     'targets': torch.from_numpy(np.array([[[[1]*40]*5]*4], dtype=np.int32)),
# }

model = EXPTeller(model_cfgs, len(tokenizer.vocab))    # 获取网络的预测值
print(model)
y = model(batch)

vis_graph = h.build_graph(model, batch)   # 获取绘制图像的对象
vis_graph.theme = h.graph.THEMES["blue"].copy()     # 指定主题颜色
vis_graph.save("./imgs/v1")   # 保存图像的路径
