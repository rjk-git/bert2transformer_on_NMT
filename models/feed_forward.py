from mxnet.gluon import nn

from hyperParameters import GetHyperParameters as ghp


class FeedForward(nn.Block):
    def __init__(self, **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        with self.name_scope():
            self.ffn_dense = nn.Dense(ghp.ffn_dim, activation="relu", use_bias=True, flatten=False)
            self.model_dense = nn.Dense(ghp.model_dim, use_bias=True, flatten=False)
            self.dropout = nn.Dropout(ghp.ffn_dropout)
            self.layer_norm = nn.LayerNorm(axis=-1, epsilon=ghp.norm_epsilon)

    def forward(self, x, *args):
        # x shape : (batch_size, seq_len, model_dim)
        residual = x

        # output shape : (batch_size, seq_len, ffn_dim)
        output = self.ffn_dense(x)

        # output shape : (batch_size, seq_len, model_dim)
        output = self.model_dense(output)

        # shape : (batch_size, seq_len, model_dim)
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output
