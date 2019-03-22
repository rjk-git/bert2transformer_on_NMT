from mxnet import nd
from mxnet.gluon import nn
from hyperParameters import GetHyperParameters as ghp


class MultiHeadAttention(nn.Block):
    def __init__(self, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        with self.name_scope():
            self.scale_attentions = ScaledDotProductAttention()
            self.dropout = nn.Dropout(ghp.dropout)
            self.LayerNorm = nn.LayerNorm()

    def forward(self, queries, keys, values=None, mask=None):
        # 残差连接
        residual = queries
        real_batch_size = queries.shape[0]

        if values is None: values = keys
        # shape [batch_size * num_head, seq_len, dim_k]
        queries = nd.reshape(queries, shape=(real_batch_size * ghp.head_nums, 0, ghp.k_dims))
        keys = nd.reshape(keys, shape=(real_batch_size * ghp.head_nums, 0, ghp.k_dims))
        values = nd.reshape(values, shape=(real_batch_size * ghp.head_nums, 0, ghp.k_dims))
        scale = ghp.k_dims ** -0.5

        if mask is not None:
            mask = nd.expand_dims(mask, axis=0)
            mask = nd.broadcast_axes(mask, axis=0, size=ghp.head_nums)
            mask = nd.reshape(mask, shape=(real_batch_size * ghp.head_nums, ghp.max_seq_len, ghp.max_seq_len))

        # shape [batch_size * num_head, seq_len, dim_k]
        output, att = self.scale_attentions(queries, keys, values, scale, mask)
        # shape [batch_size, seq_len, dim_model]
        output = nd.reshape(output, shape=(real_batch_size, 0, ghp.model_dims))
        output = self.dropout(output)
        output = self.LayerNorm(output + residual)
        return output, att


class ScaledDotProductAttention(nn.Block):
    def __init__(self, **kwargs):
        super(ScaledDotProductAttention, self).__init__(**kwargs)
        with self.name_scope():
            self.dropout = nn.Dropout(ghp.dropout)
            self.q_dense = nn.Dense(ghp.k_dims, use_bias=True, flatten=False)
            self.k_dense = nn.Dense(ghp.k_dims, use_bias=True, flatten=False)
            self.v_dense = nn.Dense(ghp.k_dims, use_bias=True, flatten=False)

    def forward(self, queries, keys, values=None, scale=None, mask=None):
        """
        Args:
            queries (nd.NDArray): (batch_size, q_len, dim_q)
            keys (nd.NDArray): (batch_size, k_len, dim_k)
            values (nd.NDArray): (batch_size, v_len, dim_v)

        Returns:
            output (nd.NDArray): (batch_size, q_len, dim_q)
            att (nd.NDArray): (batch_size, q_len, k_len)
        """

        # queries shape: (batch_size, query_seq_length, dim_q)
        # keys shape: (batch_size, key_seq_length, dim_k)
        # values shape: (batch_size, value_seq_length, dim_v)
        if values is None: values = keys
        queries = self.q_dense(queries)
        keys = self.k_dense(keys)
        values = self.v_dense(values)

        # att_score shape: (batch_size, query_seq_length, key_seq_length)
        att_scores = nd.batch_dot(queries, keys, transpose_b=True)

        if scale is not None:
            att_scores = att_scores * scale

        # mask shape: (batch_size, query_seq_length, key_seq_length)

        if mask is not None:
            att_scores = att_scores * mask

        att_scores = self.dropout(att_scores)

        # att_weights shape: (batch_size, query_seq_length, key_seq_length)
        att_weights = nd.softmax(att_scores)

        # output shape: (batch_size, query_seq_length, dim_v(dim_k))
        output = nd.batch_dot(att_weights, values)

        return output, att_weights