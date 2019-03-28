from mxnet import nd
from mxnet.gluon import nn
import numpy as np
from hyperParameters import GetHyperParameters as ghp


class MultiHeadAttention(nn.Block):
    def __init__(self, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        with self.name_scope():
            self.queries_dense = nn.Dense(ghp.model_dim, activation="relu", flatten=False)
            self.keys_dense = nn.Dense(ghp.model_dim, activation="relu", flatten=False)
            self.values_dense = nn.Dense(ghp.model_dim, activation="relu", flatten=False)
            self.dropout = nn.Dropout(ghp.dropout)
            self.LayerNorm = nn.LayerNorm(epsilon=1e-8)

    def forward(self, queries, keys, values=None, causality=False, is_training=True):
        # 残差连接
        residual = queries
        batch_size = queries.shape[0]
        q_len = queries.shape[1]
        k_len = keys.shape[1]

        if values is None:
            values = keys
        # queries shape: (batch_size, q_len, model_dim)
        # keys shape: (batch_size, k_len, model_dim)
        # values shape: (batch_size, k_len, model_dim)

        Q = self.queries_dense(queries)
        K = self.keys_dense(keys)
        V = self.values_dense(values)

        # c_dim = model_dim / head_num
        # queries_ shape [batch_size * head_nums, q_len, c_dim]
        # keys_ shape [batch_size * head_nums, k_len, c_dim]
        # values_ shape [batch_size * head_nums, k_len, c_dim]
        Q_ = nd.reshape(Q, shape=(batch_size * ghp.head_num, 0, ghp.c_dim))
        K_ = nd.reshape(K, shape=(batch_size * ghp.head_num, 0, ghp.c_dim))
        V_ = nd.reshape(V, shape=(batch_size * ghp.head_num, 0, ghp.c_dim))

        scale = ghp.c_dim ** -0.5

        # att_score
        # shape: (batch_size * head_nums, q_len, k_len)
        att_scores = nd.batch_dot(Q_, K_, transpose_b=True)

        # scale
        # shape: (batch_size * head_nums, q_len, k_len)
        att_scores = att_scores * scale

        # padding_mask
        # shape (batch_size, k_len, model_dim) -> shape (batch_size, k_len)
        key_masks = nd.sign(nd.abs(nd.sum(keys, axis=-1)))
        key_masks = nd.expand_dims(key_masks, axis=0)
        key_masks = nd.broadcast_axes(key_masks, axis=0, size=ghp.head_num)
        # shape (batch_size, k_len) -> shape (batch_size * head_nums, k_len)
        key_masks = nd.reshape(key_masks, shape=(batch_size * ghp.head_num, k_len))
        key_masks = nd.expand_dims(key_masks, axis=1)
        key_masks = nd.broadcast_axes(key_masks, axis=1, size=q_len)
        # shape (batch_size * head_nums, k_len) -> shape (batch_size * head_nums, q_len, k_len)
        key_masks = nd.reshape(key_masks, shape=(batch_size * ghp.head_num, q_len, k_len))
        paddings = nd.ones_like(att_scores) * ghp.epsilon
        att_scores = nd.where(nd.equal(key_masks, 0), paddings, att_scores)

        # att_weights shape: (batch_size * head_nums, q_len, k_len)
        att_weights = nd.softmax(att_scores, axis=2)

        # causality
        if causality:
            mask_matrix = np.ones(shape=(q_len, k_len), dtype=np.float)
            mask = np.tril(mask_matrix, k=0)
            mask = nd.expand_dims(nd.array(mask, ghp.ctx), axis=0)
            mask = nd.broadcast_axes(mask, axis=0, size=batch_size * ghp.head_num)
            paddings = nd.ones_like(mask) * ghp.epsilon
            att_weights = nd.where(nd.equal(mask, 0), paddings, att_weights)

        # query mask
        # shape (batch_size, q_len, model_dim) -> shape (batch_size, q_len)
        query_masks = nd.sign(nd.abs(nd.sum(queries, axis=-1)))
        query_masks = nd.expand_dims(query_masks, axis=0)
        query_masks = nd.broadcast_axes(query_masks, axis=0, size=ghp.head_num)
        # shape (batch_size, q_len) -> shape (batch_size * head_nums, q_len)
        query_masks = nd.reshape(query_masks, shape=(batch_size * ghp.head_num, q_len))
        query_masks = nd.expand_dims(query_masks, axis=-1)
        query_masks = nd.broadcast_axes(query_masks, axis=2, size=k_len)
        # shape (batch_size * head_nums, q_len) -> shape (batch_size * head_nums, q_len, k_len)
        query_masks = nd.reshape(query_masks, shape=(batch_size * ghp.head_num, q_len, k_len))
        att_weights = att_weights * query_masks

        # output
        # shape: (batch_size * head_nums, q_len, c_dim)
        output = nd.batch_dot(att_weights, V_)

        # shape [batch_size, seq_len, dim_model]
        output = nd.reshape(output, shape=(batch_size, 0, ghp.model_dim))

        if is_training:
            output = self.dropout(output)

        output = output + residual

        output = self.LayerNorm(output)

        return output

