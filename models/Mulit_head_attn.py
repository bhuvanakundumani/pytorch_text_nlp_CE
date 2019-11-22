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
        Q = self.query_tfm(queries) # (Batch, Seq, Feature)
        K = self.key_tfm(keys) # (Batch, Seq, Feature)
        V = self.value_tfm(values) # (Batch, Seq, Feature)
        # compute multiple attention weighted sums
        x = self.attn(Q, K, V)
        return x
 
class MultiHeadAttention(nn.Module):
    """The full multihead attention block"""
    def __init__(self, d_model, d_feature, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_feature = d_feature
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
        log_size(queries, "Input queries")
        x = [attn(queries, keys, values, mask=mask) # (Batch, Seq, Feature)
             for i, attn in enumerate(self.attn_heads)]
         
        # reconcatenate
        x = torch.cat(x, dim=Dim.feature) # (Batch, Seq, D_Feature * n_heads)
        log_size(x, "concatenated output")
        x = self.projection(x) # (Batch, Seq, D_Model)
        return x


from enum import IntEnum
class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2
 
class ScaledDotProductAttention(nn.Module):
    def __init__(cself, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
 
    def forward(self, q, k, v, mask=None):
        d_k = k.size(-1) # get the size of the key
        assert q.size(-1) == d_k
 
        # compute the dot product between queries and keys for
        # each batch and position in the sequence
        attn = torch.bmm(q, k.transpose(Dim.seq, Dim.feature)) # (Batch, Seq, Seq)
        # we get an attention score between each position in the sequence
        # for each batch
 
        # scale the dot products by the dimensionality (see the paper for why we do this!)
        attn = attn / math.sqrt(d_k)
        # normalize the weights across the sequence dimension
        # (Note that since we transposed, the sequence and feature dimensions are switched)
        attn = torch.exp(attn)
        # fill attention weights with 0s where padded
        if mask is not None: attn = attn.masked_fill(mask, 0)
        attn = attn / attn.sum(-1, keepdim=True)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v) # (Batch, Seq, Feature)
        return output