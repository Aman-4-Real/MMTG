# Multi-Modal Experience Inspired AI Creation

[![Python](https://img.shields.io/badge/Python-3.7-blue.svg)](#PyTorch)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10-green.svg)](#PyTorch)

[**Paper**](https://arxiv.org/pdf/2209.02427.pdf) |
[**Data**](https://github.com/Aman-4-Real/MMTG#Data) |
[**Hugging Face**](https://huggingface.co/Aman/MMTG)

This repository contains the source code and datasets for the ACM MM 2022 paper [Multi-Modal Experience Inspired AI Creation](https://arxiv.org/pdf/2209.02427.pdf) by Cao et al.


# Abstract
AI creation, such as poem or lyrics generation, has attracted increasing attention from both industry and academic communities, with many promising models proposed in the past few years. Existing methods usually estimate the outputs based on single and independent visual or textual information. However, in reality, humans usually make creations according to their experiences, which may involve different modalities and be sequentially correlated. To model such human capabilities, in this paper, we define and solve a novel AI creation problem based on human experiences. 
<details> <summary> More (Click me) </summary> More specifically, we study how to generate texts based on sequential multi-modal information. Compared with the previous works, this task is much more difficult because the designed model has to well understand and adapt the semantics among different modalities and effectively convert them into the output in a sequential manner. To alleviate these difficulties, we firstly design a multi-channel sequence-to-sequence architecture equipped with a multi-modal attention network. For more effective optimization, we then propose a curriculum negative sampling strategy tailored for the sequential inputs. To benchmark this problem and demonstrate the effectiveness of our model, we manually labeled a new multi-modal experience dataset. With this dataset, we conduct extensive experiments by comparing our model with a series of representative baselines, where we can demonstrate significant improvements in our model based on both automatic and human-centered metrics.
</details>

# Before You Start
- Please note that this is a work done for AI creation in **Chinese**, thus the following dataset and model checkpoints are all in Chinese. However, we have tried our model training on the English data, which is constructed on English poems in the same way with our proposed pipeline, and received the same good generated results. You can try to construct some English data (based on English corpora like poems and English text-image datasets like [MovieNet](https://movienet.github.io/)) and adapt to your own domain if necessary.
- Some parts of our work are based on the large-scale Chinese multimodal pre-trained model [WenLan (a.k.a. BriVL)](https://arxiv.org/abs/2103.06561). Please refer to [this repo](https://github.com/chuhaojin/WenLan-api-document) for more information of usage. For the English version, you can replace the WenLan with OpenAI CLIP or other multimodal representation model (more details in our paper).


# Setup
Create a new virtual environment:
```
$ git clone https://github.com/Aman-4-Real/MMTG.git
$ cd MMTG/
$ conda create -n mmtg python=3.7
$ conda activate mmtg
```
Install the Python packages. Change the cudatoolkit version according to your environment if necessary.
```
$ conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
$ pip install -r requirements.txt
```


# Download
Here are the resources, which you can download at [Hugging Face](https://huggingface.co/Aman/MMTG), [GoogleDrive](https://drive.google.com/drive/folders/1y7yD6s8U7-Vm_n-G4trYdfQzZjVXXgiX?usp=sharing) or [BaiduNetDisk(0dwq)](https://pan.baidu.com/s/1_Xlfz-7MdL1gDi47EoT06g):
| FileName | Description | Path |
| - | - | - |
| \*\_data\_\*.pkl | Train, validation, and test data. | _sharing_link_/data/ |
| mmtg_ckpt.pth | The checkpoint of MMTG for your reproduction. | _sharing_link_/ckpts/ |
| GPT2_lyrics_ckpt_epoch00.ckpt | The pre-trained decoder checkpoint. It is based on GPT2 and trained on lyrics corpus. | _sharing_link_/ckpts/ |
| token_id2emb_dict.pkl | The dict file of each token in vocabulary to WenLan embeddings. | _sharing_link_/ |
## Data
The dataset used in our paper is released as follows. Due to copyright issues, we only release the visual features of the images used in our dataset. All the `.pkl` files are in list type and each item of them is in the following format:
```
{
  'topic': STRING # the topic words
  'topic_emb': LIST # embs of the topic words
  'lyrics': LIST # list of lyrics sentences
  'img_0_emb': LIST # emb of the 1st image
  'r_0': STRING # the 1st text
  'r_0_emb': LIST # emb of the 1st text
  'img_1_emb': LIST # emb of the 2nd image
  'r_1': STRING # the 2nd text
  'r_1_emb': LIST # emb of the 2nd text
  ...,
  'img_4_emb': LIST # emb of the 4th image
  'r_4': STRING # the 4th text
  'r_4_emb': LIST # emb of the 4th text
  'rating': INT # the sample level (range from 1 to 5, 5 refers to the most positive one while 1 refers to the least).
}
```
For the test data, there are additional keys:
```
{
  'score_0': {
    'img_rel': [2, 2], # the relevance score of the 1st image and the 1st & 2nd lyrics sentences (range from 1 to 5).
    'r_rel': [1, 1], # the relevence score of the 1st text and the 1st & 2nd lyrics sentences (range from 1 to 5).
    'cmp_rel': [0, 0] # whether the image or the text is more relevant to the lyrics. 0 refers to the image and 2 refers to the text (1 means a tie).
  } # a list above means: [rator1_score, rator2_score]
  ...,
  'score_4': ...
}
```
You can use this additional labeled information to analyze your parameters (like attention weights) and results.

## Checkpoints
`mmtg_ckpt.pth`: The checkpoint of MMTG for your reproduction. It is trained on the dataset we released. You can simply load it and use it to generate on your own data or for the demo.

`GPT2_lyrics_ckpt_epoch00.ckpt`: The pre-trained decoder checkpoint. As mentioned in our paper, we use a pre-trained GPT2 to initialize our decoder and fine-tune it on our lyrics corpus (phase 1). While doing the whole training (phase 2), we start from this fine-tuned one.

## Other
`token_id2emb_dict.pkl`: The dict file of each token in vocabulary to WenLan embeddings. It is used to convert the token ids to the corresponding embeddings in phase 1 and phase 2. This is to adapt the text embedding space to the image embedding space. You can also use other pre-trained multimodal representation models (like OpenAI CLIP) to replace WenLan and construct an English one.


# Usage
1. Download the `data files`, `pre-trained GPT2 checkpoint`, and `token_id2emb_dict.pkl`.
2. Put them in `./data/`, `./src/pretrained/` (change the path in `./src/configs.py` correspondingly) and `./src/vocab/` respectively.

## Training
Change your configs and run:
```
$ cd src/
$ bash train.sh
```

## Generate
Change your configs and run:
```
$ cd src/
$ bash generate.sh
```
This will generate the results of the test data and save them in your `save_samples_path`. You can also use the checkpoint we released to generate on your own data. The format of the data is the same as the test data (without the scores and ratings). You can refer to `./data/test_data.pkl` for more details.

<!-- ## Demo
We provide a demo to easily visualize the input and the output. You can run:
```
$ cd src/demo/
$ python main.py
```
Then go to the interactive and more user-friendly page and enjoy! -->


# Citation
If you find this paper and repo useful, please cite us in your work:

<!--
```
@article{cao2022multi,
  title={Multi-Modal Experience Inspired AI Creation},
  author={Cao, Qian and Chen, Xu and Song, Ruihua and Jiang, Hao and Yang, Guang and Cao, Zhao},
  journal={arXiv preprint arXiv:2209.02427},
  year={2022}
}
```
-->

```
@inproceedings{10.1145/3503161.3548189,
  author = {Cao, Qian and Chen, Xu and Song, Ruihua and Jiang, Hao and Yang, Guang and Cao, Zhao},
  title = {Multi-Modal Experience Inspired AI Creation},
  year = {2022},
  isbn = {9781450392037},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3503161.3548189},
  doi = {10.1145/3503161.3548189},
  booktitle = {Proceedings of the 30th ACM International Conference on Multimedia},
  pages = {1445–1454},
  numpages = {10},
  keywords = {AI creation, multi-modal, experience},
  location = {Lisboa, Portugal},
  series = {MM '22}
}
```
For any questions, please feel free to reach me at caoqian4real@ruc.edu.cn.





