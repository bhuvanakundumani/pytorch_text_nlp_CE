import nltk
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F 

class LstmAttentionModel(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length,pretrained_flag, pretrained_embed ):
        super(LstmAttentionModel, self).__init__()
        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length
        if pretrained_flag:
            self.embed = nn.Embedding.from_pretrained(pretrained_embed, freeze=False)
        else:
            self.embed = nn.Embedding(self.vocab_size, self.embedding_length)
        self.attn_weights = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.lstm = nn.LSTM(embedding_length, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)


    def attention_net(self, output_lstm, final_hidden_state):
        import ipdb;ipdb.set_trace()
        hidden = final_hidden_state.squeeze(0).unsqueeze(2)
        bsize, hsize, _ = hidden.size()
        v = self.attn_weights.size()
        attended_hidden = torch.bmm( self.attn_weights.expand( bsize,*v),hidden)      
        attn_weights = torch.bmm( output_lstm,  attended_hidden).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights,1)
        new_hidden_state = torch.bmm(output_lstm.transpose(1,2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return new_hidden_state

    def forward(self, x):
        h_embedding = self.embed(x)
        import ipdb; ipdb.set_trace();
        output, (h_lstm, c_lstm) = self.lstm(h_embedding)
        attn_output = self.attention_net(output, h_lstm)
        logits = self.out(attn_output)
        #print(logits.shape)
        return logits

    '''

    #
    def attention_net(self, output_lstm, final_hidden_state):
        import ipdb;ipdb.set_trace()
        hidden = final_hidden_state.squeeze(0).unsqueeze(2)
        bsize, hsize, _ = hidden.size()
        v = self.attn_weights.size()
        attended_hidden = torch.bmm( self.attn_weights.expand( bsize,*v),hidden)
        
        attn_weights = torch.bmm( output_lstm,  attended_hidden).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights,1)
        new_hidden_state = torch.bmm(output_lstm.transpose(1,2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return new_hidden_state

    

    def attention_net(self, output_lstm, final_hidden_state):
        hidden = final_hidden_state.squeeze(0) 
        attn_weights = torch.bmm(output_lstm, hidden.unsqueeze(2)).squeeze(2)
        soft_attn_weights = F.softmax(attn_weights,1)
        new_hidden_state = torch.bmm(output_lstm.transpose(1,2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return new_hidden_state

    '''

        