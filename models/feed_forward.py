from mxnet import nd
from mxnet.gluon import nn
from hyperParameters import GetHyperParameters as ghp


class FeedForward(nn.Block):
    def __init__(self, **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        with self.name_scope():
            self.w1 = nn.Conv1D(in_channels=ghp.model_dims, channels=ghp.ffn_dims, kernel_size=1)
            self.w2 = nn.Conv1D(in_channels=ghp.ffn_dims, channels=ghp.model_dims, kernel_size=1)
            self.dropout = nn.Dropout(ghp.dropout)
            self.layer_norm = nn.LayerNorm()

    def forward(self, x, *args):
        # x shape : (batch_size, seq_len, dim_model)
        residual = x

        # shape : (batch_size, dim_model, seq_len)
        output = nd.transpose(x, axes=(0, 2, 1))

        # shape : (batch_size, dim_model, seq_len)
        output = self.w2(nd.relu(self.w1(output)))

        # shape : (batch_size, seq_len, dim_model)
        output = nd.transpose(output, axes=(0, 2, 1))

        # shape : (batch_size, seq_len, dim_model)
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output
