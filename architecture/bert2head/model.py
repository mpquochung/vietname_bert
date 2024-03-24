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

class CNN2HEAD_UIT(Module):
    def __init__(self, embedding_dim, n_filters=96, filter_sizes= [1,2,3,5],  dropout=0.2,activation=None):

        super().__init__()
        self.ln1= nn.LayerNorm(embedding_dim)
        self.fc_input = nn.Linear(embedding_dim, embedding_dim)

        self.conv_0_0 = nn.Conv1d(in_channels = embedding_dim,
                                out_channels = n_filters,
                                  padding='same',
                                kernel_size = filter_sizes[0])
        self.conv_1_0 = nn.Conv1d(in_channels = embedding_dim,
                                out_channels = n_filters,
                                  padding='same',
                                kernel_size = filter_sizes[1])
        self.conv_2_0 = nn.Conv1d(in_channels = embedding_dim,
                                out_channels = n_filters,
                                  padding='same',
                                kernel_size = filter_sizes[2])
        self.conv_3_0 = nn.Conv1d(in_channels = embedding_dim,
                                out_channels = n_filters,
                                  padding='same',
                                kernel_size = filter_sizes[3])
        
        self.activation = nn.SiLU() if activation is None else activation
        
        # self.linear_sent = nn.Linear(8192, 256)
        # self.linear_clas = nn.Linear(8192, 256)
        self.dropout_1 = nn.Dropout(0.2)

        self.out_sent = nn.Linear(n_filters*len(filter_sizes),4)
        self.out_clas = nn.Linear(n_filters*len(filter_sizes),10)

        # self.dropout_sent = nn.Dropout(dropout)
        # self.dropout_clas = nn.Dropout(dropout)
    def forward(self, encoded):
        encoded = encoded.to(device)
        embedded = self.activation(self.fc_input(encoded)) # bs, 64, 768

        # Warmup and create general feature selection before divide into 2 heads 
        embedded = self.ln1(embedded)
        embedded = embedded.permute(0, 2, 1) # bs, 768, 64

        conved_0_0 = self.activation(self.conv_0_0(embedded))
        conved_1_0 = self.activation(self.conv_1_0(embedded))
        conved_2_0 = self.activation(self.conv_2_0(embedded))
        conved_3_0 = self.activation(self.conv_3_0(embedded))

        pooled_0_0 = F.max_pool1d(conved_0_0, conved_0_0.shape[2]).squeeze(2) # bs, 64, 32
        pooled_1_0 = F.max_pool1d(conved_1_0, conved_1_0.shape[2]).squeeze(2)
        pooled_2_0 = F.max_pool1d(conved_2_0, conved_2_0.shape[2]).squeeze(2)
        pooled_3_0 = F.max_pool1d(conved_3_0, conved_3_0.shape[2]).squeeze(2)
        cat= self.dropout_1(torch.cat((pooled_0_0, pooled_1_0, pooled_2_0, pooled_3_0), dim = 1))
        # bs, 256

        # Sentiment Head
        sent = self.out_sent(cat)

        # Classification Head
        clas = self.out_clas(cat)

        return sent, clas


class BertCNN2HEAD_UIT(Module):
    def __init__(self, name, loss = "genloss"):
        super().__init__()
        self.BertModel = AutoModel.from_pretrained(name)
        self.BertModel = self.BertModel.to(device)
        self.cnn = CNN2HEAD_UIT(768)
        self.cnn = self.cnn.to(device)
    def forward(self,sentences,attention):
      embedded = self.BertModel(sentences,attention_mask=attention)[0]
      sent,clas = self.cnn(embedded)
      return sent,clas

# @title Finetune Model

class Linear2HEAD(Module):
    def __init__(self, embedding_dim, dropout=0.2,activation=None):

        super().__init__()
        self.ln1= nn.LayerNorm(embedding_dim)
        # self.ln2= nn.LayerNorm(len(filter_sizes)*n_filters)
        self.fc_input = nn.Linear(embedding_dim, embedding_dim)

        self.activation = nn.SiLU() if activation is None else activation


        self.out_sent = nn.Linear(embedding_dim,4)
        self.out_clas = nn.Linear(embedding_dim,10)

        self.dropout_sent = nn.Dropout(dropout)
        self.dropout_clas = nn.Dropout(dropout)
    def forward(self, encoded):

        embedded = self.activation(self.fc_input(encoded)) # bs, 64, 768

        # Warmup truoc khi chia head
        embedded = self.ln1(embedded)
       
        sent = self.out_sent(embedded)
        clas = self.out_clas(embedded)

        return sent, clas


class BertLinear2HEAD(Module):
    def __init__(self, name):
        super().__init__()
        self.BertModel = AutoModel.from_pretrained(name)
        self.BertModel.to(device)
        # self.phoBertModel.load_state_dict(torch.load(pretrained_path))
        self.linear = Linear2HEAD(768)
    def forward(self,sentences,attention):
       embedded = self.BertModel(sentences,attention_mask=attention).last_hidden_state[:,0,:]
       sent,clas = self.linear(embedded)
       return sent,clas
    

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

        # Warmup truoc khi chia head
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
    def forward(self,sentences,attention,sentences2=None, mlm=False):
       embedded = self.BertModel(sentences,attention_mask=attention).last_hidden_state[:,0,:]
       embedded2 = nn.Identity(768)
       if mlm:
         embedded2 = self.BertModel(sentences2,attention)[0]
       return self.linear(embedded,embedded2, mlm=mlm)