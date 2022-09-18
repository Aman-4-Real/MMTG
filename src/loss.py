import torch
import torch.nn as nn

# class MyNLLLoss(torch.nn.Module):
#     def __init__(self):
#         super(MyNLLLoss, self).__init__()

#     def nll(self, y, p):
#         NEAR_0 = 1e-10
#         p = torch.exp(torch.sum(torch.log(p)))
#         return - y * torch.log(p + NEAR_0) - (1 - y) * torch.log(1 - p + NEAR_0)

#     def forward(self, preds, targets, ratings, stage):
#         '''
#         Args:
#             preds: (batch_size, seq_len, _max_sent_length*2+2, vocab_size)
#             targets: (batch_size, seq_len, _max_sent_length*2+2)
#             ratings: (batch_size)
#         '''
#         import pdb; pdb.set_trace()
#         device = preds.device
#         zero = torch.zeros_like(ratings)
#         one = torch.ones_like(ratings)
#         if stage == 1:
#             ratings = torch.where(ratings > 4, one, zero)
#         else:
#             ratings = torch.where(ratings > 3, one, zero)
#         preds = nn.Softmax(dim=-1)(preds)
#         loss = torch.zeros(preds.shape[0]).to(device)
#         for i in range(preds.shape[0]): # batch_size
#             y = ratings[i]
#             for j in range(preds.shape[1]): # seq_len
#                 token_probs = preds[i,j,torch.arange(preds.shape[2]),targets[i][j]]
#                 loss[i] += self.nll(y, token_probs)
        
#         return torch.mean(loss/preds.shape[1])


class MyLoss(torch.nn.Module):
    def __init__(self, data_config, model_cfgs):
        super(MyLoss, self).__init__()
        self._max_topic_len = data_config.topic_prompt_length
        self._seq_len = model_cfgs['seq_len']

    def forward(self, outputs, targets, ratings, stage):
        '''
        Args:
            outputs: (batch_size, topic_prompt_length + max_seq_length + 1, vocab_size)
            targets: (batch_size, max_seq_length + 1)
            ratings: (batch_size)
        '''
        NEAR_0 = 1e-10
        device = outputs.device
        zero = torch.zeros_like(ratings)
        one = torch.ones_like(ratings)
        batch_size = targets.shape[0]
        if stage == 1:
            ratings = torch.where(ratings > 4, one, zero)
        else:
            ratings = torch.where(ratings > 3, one, zero)

        shift_logits = outputs[:, self._max_topic_len:-1, :]
        shift_labels = targets[:, 1:]
        
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss()
        
        loss = torch.zeros(batch_size).to(device)
        for i in range(batch_size):
            y = ratings[i]
            _loss = loss_fct(shift_logits[i], shift_labels[i])
            p = 1/torch.exp(_loss)
            loss[i] += torch.sum(- y * torch.log(p + NEAR_0) - (1 - y) * torch.log(1 - p + NEAR_0))
        return torch.mean(loss)
         