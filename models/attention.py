from mxnet import nd
from mxnet.gluon import nn
import numpy as np
from hyperParameters import GetHyperParameters as ghp


class MultiHeadAttention(nn.Block):
    def __init__(self, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        with self.name_scope():
            self.queries_dense = nn.Dense(ghp.model_dim, use_bias=False, flatten=False)
            self.keys_dense = nn.Dense(ghp.model_dim, use_bias=False, flatten=False)
            self.values_dense = nn.Dense(ghp.model_dim, use_bias=False, flatten=False)
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
        # print("keys", keys)
        Q = self.queries_dense(queries)
        K = self.keys_dense(keys)
        V = self.values_dense(values)
        # print("Q", Q)
        # print("K", K)
        # c_dim = model_dim / head_num
        # queries_ shape [batch_size * head_nums, q_len, c_dim]
        # keys_ shape [batch_size * head_nums, k_len, c_dim]
        # values_ shape [batch_size * head_nums, k_len, c_dim]
        Q_ = nd.transpose(Q.reshape(shape=(0, 0, ghp.head_num, -1)),
                          axes=(0, 2, 1, 3)).reshape(shape=(-1, 0, 0), reverse=True)
        K_ = nd.transpose(K.reshape(shape=(0, 0, ghp.head_num, -1)),
                          axes=(0, 2, 1, 3)).reshape(shape=(-1, 0, 0), reverse=True)
        V_ = nd.transpose(V.reshape(shape=(0, 0, ghp.head_num, -1)),
                          axes=(0, 2, 1, 3)).reshape(shape=(-1, 0, 0), reverse=True)
        # print("Q_", Q_)
        # print("K_", K_)
        # batch01: (seq01_len, c_dim01)
        # batch02: (seq01_len, c_dim02)
        # batch03: (seq01_len, c_dim03)
        # ...

        scale = ghp.c_dim ** -0.5

        # att_score
        # shape: (batch_size * head_nums, q_len, k_len)
        att_scores = nd.batch_dot(Q_, K_, transpose_b=True)
        # print("att_scores", att_scores)
        # scale
        # shape: (batch_size * head_nums, q_len, k_len)
        att_scores = att_scores * scale
        # print("att_scores_scale", att_scores)

        #############
        # keys_mask #
        #############
        # shape (batch_size, k_len, model_dim) -> shape (batch_size, k_len)
        key_masks = nd.sign(nd.abs(nd.sum(keys, axis=-1)))

        # shape (batch_size, k_len) -> shape (batch_size, q_len, k_len)
        key_masks = nd.expand_dims(key_masks, axis=1)
        key_masks = nd.broadcast_axes(key_masks, axis=1, size=q_len)

        # shape (batch_size, q_len, k_len) -> shape (batch_size * head_nums, q_len, k_len)
        key_masks = nd.expand_dims(key_masks, axis=1)
        key_masks = nd.broadcast_axes(key_masks, axis=1, size=ghp.head_num)
        key_masks = nd.reshape(key_masks, shape=(batch_size * ghp.head_num, q_len, k_len))
        # print("key_masks", key_masks)

        padding = nd.zeros_like(att_scores)
        att_scores = nd.where(nd.equal(key_masks, 0), padding, att_scores)
        # print("att_scores_key_masks", att_scores)
        # att_weights shape: (batch_size * head_nums, q_len, k_len)
        att_weights = nd.softmax(att_scores, axis=2)
        # print("att_weights", att_weights)

        #############
        # causality #
        #############
        if causality:
            mask_matrix = np.ones(shape=(q_len, k_len), dtype=np.float)
            mask = np.tril(mask_matrix, k=0)
            mask = nd.expand_dims(nd.array(mask, ghp.ctx), axis=0)
            mask = nd.broadcast_axes(mask, axis=0, size=batch_size * ghp.head_num)
            # paddings = nd.zeros_like(mask)
            # att_weights = nd.where(nd.equal(mask, 0), paddings, att_weights)
            # print("causality_mask", mask)
            att_weights = att_weights * mask
            # print("att_weights_causality", att_weights)

        ##############
        # query mask #
        ##############
        # shape (batch_size, q_len, model_dim) -> shape (batch_size, q_len)
        query_masks = nd.sign(nd.abs(nd.sum(queries, axis=-1)))

        # shape (batch_size, q_len) -> shape (batch_size, k_len, q_len)
        query_masks = nd.expand_dims(query_masks, axis=1)
        query_masks = nd.broadcast_axes(query_masks, axis=1, size=k_len)

        # shape (batch_size, k_len, q_len) -> shape (batch_size, q_len, k_len)
        query_masks = nd.transpose(query_masks, axes=(0, 2, 1))

        # shape (batch_size, q_len, k_len) -> shape (batch_size * head_nums, q_len, k_len)
        query_masks = nd.expand_dims(query_masks, axis=1)
        query_masks = nd.broadcast_axes(query_masks, axis=1, size=ghp.head_num)
        query_masks = nd.reshape(query_masks, shape=(batch_size * ghp.head_num, q_len, k_len))
        # print("query_mask", query_masks)
        att_weights = att_weights * query_masks
        # print("att_weights_query_mask", att_weights)
        # output
        # shape: (batch_size * head_nums, q_len, c_dim)
        # print("V_", V_)
        output = nd.batch_dot(att_weights, V_)
        # print("output", output)
        # shape [batch_size, seq_len, dim_model]
        # wrong write like this -> output = nd.reshape(output, shape=(batch_size, 0, ghp.model_dim))
        output = nd.reshape(output, shape=(batch_size, ghp.head_num, ghp.max_seq_len, ghp.c_dim))
        output = nd.transpose(output, axes=(0, 2, 1, 3))
        output = nd.reshape(output, shape=(batch_size, ghp.max_seq_len, ghp.model_dim))
        # print("output_0sel", output[0])
        if is_training:
            output = self.dropout(output)

        output = output + residual

        output = self.LayerNorm(output)

        return output
