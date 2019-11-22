from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# TODO - check with K if unsqueeze() the same as tf.newaxis
def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq,0), tf.float32)
    # add extra dimensions to add to the paddign to the attention logits
    return seq[:, tf.newaxis, tf.newaxis, :] # batchsize, 1,1 ,seq_length

def scaled_dot_product_attention(q, k, v, mask):
    #import ipdb;ipdb.set_trace();
    matmul_qk = tf.matmul(q, k, transpose_b = True)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attn_logits = matmul_qk /tf.math.sqrt(dk)

    # adding mask to the scaled tensor
    if mask is not None:
        scaled_attn_logits += (mask * -1e9)
        # softmax normalised on the last axis seq_len_k
    attention_weights = tf.nn.softmax(scaled_attn_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights


class MutliHeadAtention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MutliHeadAtention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])


    def call(self, v, k, q, mask):
        import ipdb; ipdb.set_trace();
        batch_size = tf.shape(q)[0]
        q = self.wq(q) # batch_size, seq_len, d_model
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        print("shape of q is ",q.shape)
        print("shape of k is", k.shape)
        print("shape of v is ", v.shape) 

        #scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        #attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)

        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, 
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        
        return output, attention_weights

#testing 

temp_mha = MutliHeadAtention(d_model=512, num_heads=8)
y = tf.random.uniform((1, 60, 512))  # (batch_size, encoder_sequence, d_model)
out, attn = temp_mha(y, k=y, q=y, mask=None)
print(f" Output shape is {out.shape}  and shape of attention is  {attn.shape}")
   

# point wise feedforward network 
