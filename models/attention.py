import numpy as np

from mxnet import nd
from mxnet.gluon import nn

from hyperParameters import GetHyperParameters as ghp


class MultiHeadAttention(nn.Block):
    def __init__(self, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        with self.name_scope():
            self.queries_dense = nn.Dense(ghp.model_dim, use_bias=False, flatten=False)
            self.keys_dense = nn.Dense(ghp.model_dim, use_bias=False, flatten=False)
            self.values_dense = nn.Dense(ghp.model_dim, use_bias=False, flatten=False)
            self.dropout = nn.Dropout(ghp.dropout)
            self.LayerNorm = nn.LayerNorm(epsilon=ghp.norm_epsilon)

    def forward(self, queries, keys, values, mask, is_training=True):
        # queries shape: (batch_size, q_len, model_dim)
        # keys shape: (batch_size, k_len, model_dim)
        # values shape: (batch_size, k_len, model_dim)

        # residual
        residual = queries
        batch_size = queries.shape[0]
        q_len = queries.shape[1]
        k_len = keys.shape[1]

        Q = self.queries_dense(queries)
        K = self.keys_dense(keys)
        V = self.values_dense(values)

        c_dim = int(ghp.model_dim / ghp.head_num)

        # Qs(list) shape (batch_size, q_len, c_dim) * head_num
        # Ks(list) shape (batch_size, k_len, c_dim) * head_num
        # Vs(list) shape (batch_size, k_len, c_dim) * head_num
        Qs = nd.split(Q, num_outputs=ghp.head_num, axis=-1)
        Ks = nd.split(K, num_outputs=ghp.head_num, axis=-1)
        Vs = nd.split(V, num_outputs=ghp.head_num, axis=-1)

        # Q_ shape (batch_size * num_head, q_len, c_dim)
        Q_ = nd.empty(shape=(1, q_len, c_dim), ctx=ghp.ctx)
        for q in Qs:
            Q_ = nd.concat(Q_, q, dim=0)
        Q_ = Q_[1:]

        # K_ shape (batch_size * num_head, k_len, c_dim)
        K_ = nd.empty(shape=(1, k_len, c_dim), ctx=ghp.ctx)
        for k in Ks:
            K_ =  nd.concat(K_, k, dim=0)
        K_ = K_[1:]

        # V_ shape (batch_size * num_head, k_len, c_dim)
        V_ = nd.empty(shape=(1, k_len, c_dim), ctx=ghp.ctx)
        for v in Vs:
            V_ = nd.concat(V_, v, dim=0)
        V_ = V_[1:]

        # batch01: (seq01_len, c_dim01)
        # batch02: (seq02_len, c_dim01)
        # batch03: (seq03_len, c_dim01)
        # ...
        # batch09: (seq01_len, c_dim02)
        # ...

        scale = ghp.c_dim ** -0.5

        # att_score
        # shape: (batch_size * head_num, q_len, k_len)
        att_scores = nd.batch_dot(Q_, K_, transpose_b=True)

        # scale
        # shape: (batch_size * head_num, q_len, k_len)
        att_scores = att_scores * scale

        # mask
        mask = nd.expand_dims(mask, axis=0)
        mask = nd.broadcast_axes(mask, axis=0, size=ghp.head_num)
        mask = nd.reshape(mask, shape=(-1, q_len, k_len))
        padding = nd.ones_like(mask) * -np.inf
        att_scores = nd.where(nd.equal(mask, 0), padding, att_scores)

        # att_weights shape: (batch_size * head_num, q_len, k_len)
        att_weights = nd.softmax(att_scores, axis=-1)
        output = nd.batch_dot(att_weights, V_)

        outputs = nd.split(output, num_outputs=ghp.head_num, axis=0)
        empty = nd.empty(shape=(batch_size, q_len, 1), ctx=ghp.ctx)
        for out in outputs:
            empty = nd.concat(empty, out, dim=-1)
        output = empty[:, :, 1:]

        if is_training:
            output = self.dropout(output)

        output = output + residual
        output = self.LayerNorm(output)
        return output
