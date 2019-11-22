import nltk
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F 
import math


class LstmMultiAttentionModel(nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embedding_length,pretrained_flag, pretrained_embed ):
        super(LstmMultiAttentionModel, self).__init__()
        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length
        if pretrained_flag:
            self.embed = nn.Embedding.from_pretrained(pretrained_embed, freeze=False)
        else:
            self.embed = nn.Embedding(self.vocab_size, self.embedding_length)
        self.attention_heads = MultiHeadAttention(hidden_size, 25, n_heads = 4)
        self.lstm = nn.LSTM(embedding_length, hidden_size, batch_first=True)
        self.proj = nn.Linear(hidden_size * 256,hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # import ipdb;ipdb.set_trace();
        h_embedding = self.embed(x)
        output, (h_lstm, c_lstm) = self.lstm(h_embedding)
        #attn_output = self.attention_net(output, h_lstm)  # THIS IS WHERE THE MULTI HEADED ATTENTION GETS CALLED     
        attn_output = self.attention_heads(output, output, output)
        logits = self.out(attn_output)
        #print(logits.shape)
        return logits

class MultiHeadAttention(nn.Module):
    """The full multihead attention block"""
    def __init__(self, d_model, d_feature, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model #hidden dimension
        self.d_feature = d_feature # since we need the input and output features for nn.Linear - we give d_feature = d_model divided by n_heads
        self.n_heads = n_heads
        # in practice, d_model == d_feature * n_heads
        assert d_model == d_feature * n_heads

        # Note that this is very inefficient:
        # I am merely implementing the heads separately because it is 
        # easier to understand this way
        self.attn_heads = nn.ModuleList([
            AttentionHead(d_model, d_feature, dropout) for _ in range(n_heads)
        ])
        self.projection = nn.Linear(d_feature * n_heads, d_model) 
    
    def forward(self, queries, keys, values, mask=None):
        #import ipdb;ipdb.set_trace();
        #log_size(queries, "Input queries")
        x = [attn(queries, keys, values, mask=mask) # (Batch, Seq, Feature)
            for i, attn in enumerate(self.attn_heads)]
        
        # reconcatenate
        x = torch.cat(x, 2) # (Batch, Seq, D_Feature * n_heads)
        #log_size(x, "concatenated output")
        x = self.projection(x) # (Batch, Seq, D_Model)
        return x

class AttentionHead(nn.Module):
    """A single attention head"""
    def __init__(self, d_model, d_feature, dropout=0.1):
        super().__init__()
        # We will assume the queries, keys, and values all have the same feature size
        self.attn = ScaledDotProductAttention(dropout)
        self.query_tfm = nn.Linear(d_model, d_feature)
        self.key_tfm = nn.Linear(d_model, d_feature)
        self.value_tfm = nn.Linear(d_model, d_feature)

    def forward(self, queries, keys, values, mask=None):
        #import ipdb; ipdb.set_trace();
        Q = self.query_tfm(queries) # (Batch, Seq, Feature)
        K = self.key_tfm(keys) # (Batch, Seq, Feature)
        V = self.value_tfm(values) # (Batch, Seq, Feature)
        # compute multiple attention weighted sums
        x = self.attn(Q, K, V)
        return x

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k = k.size(-1) # get the size of the key
        assert q.size(-1) == d_k
        #import ipdb; ipdb.set_trace();
        # compute the dot product between queries and keys for
        # each batch and position in the sequence
        attn = torch.bmm(q, k.transpose(1,2)) # (Batch, Seq, Seq)
        # we get an attention score between each position in the sequence
        # for each batch

        # scale the dot products by the dimensionality (see the paper for why we do this!)
        attn = attn / math.sqrt(d_k)
        # normalize the weights across the sequence dimension
        # (Note that since we transposed, the sequence and feature dimensions are switched)
        #attn = torch.exp(attn)
        # fill attention weights with 0s where padded
        if mask is not None: attn = attn.masked_fill(mask, 0)
        attn = attn / attn.sum(-1, keepdim=True)
        attn = self.dropout(attn)
        #import ipdb; ipdb.set_trace();
        output = torch.bmm(attn, v) # (Batch, Seq, Feature)
        return output









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

        