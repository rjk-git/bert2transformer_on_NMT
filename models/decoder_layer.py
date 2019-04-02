from mxnet.gluon import nn
from models.attention import MultiHeadAttention
from models.feed_forward import FeedForward
from mxnet import nd
from hyperParameters import GetHyperParameters as ghp


class DecoderLayer(nn.Block):
    def __init__(self, **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)

        with self.name_scope():
            self.self_masked_attention = MultiHeadAttention()
            self.context_attention = MultiHeadAttention()
            self.feed_forward = FeedForward()

    def forward(self, en_emb, zh_emb, is_training):
        # self attention, all inputs are decoder inputs
        dec_output = self.self_masked_attention(
            zh_emb,
            zh_emb,
            zh_emb,
            True, # if causality, the future info will not be compute
            is_training)

        # context attention
        # query is decoder's outputs, key and value are encoder's inputs
        query_masks = nd.broadcast_not_equal(zh_emb, nd.zeros_like(zh_emb, ctx=ghp.ctx))

        dec_output = dec_output * query_masks
        dec_output = self.context_attention(
            dec_output,
            en_emb,
            en_emb,
            False,
            is_training)

        # decoder's output, or context
        dec_output = self.feed_forward(dec_output)

        return dec_output

