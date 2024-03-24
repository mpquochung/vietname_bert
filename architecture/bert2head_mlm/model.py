import torch
from torch.nn import Module 
import torch.nn as nn
from transformers import AutoModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

"""
    Read the doc on bert3head_mlm/model.py
"""

class Linear1HEADMLM(Module):
    def __init__(self, embedding_dim, dropout=0.2,activation=None):

        super().__init__()
        self.ln1= nn.LayerNorm(embedding_dim)
        self.ln2= nn.LayerNorm(embedding_dim)
        # self.ln2= nn.LayerNorm(len(filter_sizes)*n_filters)
        self.fc_input = nn.Linear(embedding_dim, embedding_dim)
        self.fc_input2 = nn.Linear(embedding_dim, embedding_dim)
        
        self.activation = nn.SiLU() if activation is None else activation


        self.out_sent = nn.Linear(embedding_dim,4)
        self.MLM = nn.Linear(embedding_dim,64001)

    def forward(self, encoded,encoded2=None, mlm=False):
        embedded = self.activation(self.fc_input(encoded)) # bs, 64, 768
        # Warmup before head division
        embedded = self.ln1(embedded)
       
        sent = self.out_sent(embedded)
        if mlm:
          embedded2 = self.activation(self.fc_input2(encoded2))
          # New
          embedded2 = self.ln2(embedded2)
          mlm = self.MLM(embedded2)
          return sent, mlm

        return sent


class BertLinear1HEADMLM(Module):
    def __init__(self, name):
        super().__init__()
        self.BertModel = AutoModel.from_pretrained(name)
        self.BertModel.to(device)
        # self.phoBertModel.load_state_dict(torch.load(pretrained_path))
        self.linear = Linear1HEADMLM(768)
    # def forward(self,sentences,attention,sentences2=None, mlm=False):
    #    embedded = self.BertModel(sentences,attention_mask=attention).last_hidden_state[:,0,:]
    #    embedded2 = nn.Identity(768)
    #    if mlm:
    #      embedded2 = self.BertModel(sentences2,attention)[0]
    #    return self.linear(embedded,embedded2, mlm=mlm)
    
    # New forward

    def forward(self,sentences,attention,sentences2=None, mlm=False):
       embedded = self.BertModel(sentences,attention_mask=attention)
       embedded_task = embedded.last_hidden_state[:,0,:]
       embedded_mlm = nn.Identity(768)
       if mlm:
           embedded_mlm = embedded[0]
        #  embedded2 = self.BertModel(sentences2,attention)[0]
       return self.linear(embedded_task,embedded_mlm, mlm=mlm)