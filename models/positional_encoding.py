from mxnet import nd
from mxnet.gluon import nn
from hyperParameters import GetHyperParameters as ghp


class PositionalEncoding(nn.Block):
    def __init__(self, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.position_embedding = nn.Embedding(ghp.max_seq_len, ghp.model_dim)

    def forward(self, x, *args):
        zeros = nd.zeros((x.shape[0], 1), ctx=ghp.ctx)
        x = nd.broadcast_not_equal(x, zeros)
        return self.position_embedding(x)
