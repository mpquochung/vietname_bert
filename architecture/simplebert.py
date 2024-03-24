import torch
from torch.nn import Module 
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# This is the multi-task model without MLM head
# We try to also CNN for this task, but it seem to have no different with the linear model

# We use CNN architecture from `phoBERT-CNN` by a research group in UIT-NLP



class Linear1HEAD(Module):
    def __init__(self, embedding_dim, dropout=0.2,activation=None):

        super().__init__()
        self.ln1= nn.LayerNorm(embedding_dim)
        # self.ln2= nn.LayerNorm(len(filter_sizes)*n_filters)
        self.fc_input = nn.Linear(embedding_dim, embedding_dim)

        self.activation = nn.SiLU() if activation is None else activation


        self.out_sent = nn.Linear(embedding_dim,2)

        self.dropout_sent = nn.Dropout(dropout)
    def forward(self, encoded):

        embedded = self.activation(self.fc_input(encoded)) # bs, 64, 768

        # Warmup truoc khi chia head
        embedded = self.ln1(embedded)
       
        sent = self.out_sent(embedded)

        return sent


class BertLinear1HEAD(Module):
    def __init__(self, name):
        super().__init__()
        self.BertModel = AutoModel.from_pretrained(name)
        self.BertModel.to(device)
        # self.phoBertModel.load_state_dict(torch.load(pretrained_path))
        self.linear = Linear1HEAD(768)
    def forward(self,sentences,attention):
       embedded = self.BertModel(sentences,attention_mask=attention).last_hidden_state[:,0,:]
       sent = self.linear(embedded)
       return sent
    

