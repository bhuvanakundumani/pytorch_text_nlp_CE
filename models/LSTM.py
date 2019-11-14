import nltk
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F 

class LstmClassifier(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length, pretrained_flag, pretrained_embed ):
        super(LstmClassifier, self).__init__()
        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length
        if pretrained_flag:
            self.embed = nn.Embedding.from_pretrained(pretrained_embed, freeze=False)
        else:
            self.embed = nn.Embedding(vocab_size, embedding_length)
        self.lstm = nn.LSTM(embedding_length, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        #import ipdb;ipdb.set_trace();
        h_embedding = self.embed(x)
        output, (h_lstm, c_lstm) = self.lstm(h_embedding)       
        logits = self.out(h_lstm.squeeze(0))
        #logits = self.out(h_lstm.view(output.shape[0], output.shape[2]))
        return logits
        