from mxnet.gluon import nn
from models.attention import MultiHeadAttention
from models.feed_forward import FeedForward


class DecoderLayer(nn.Block):
    def __init__(self, **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)

        with self.name_scope():
            self.self_masked_attention = MultiHeadAttention()
            self.context_attention = MultiHeadAttention()
            self.feed_forward = FeedForward()

    def forward(self, en_emb, zh_emb, mask, self_mask, is_training):
        # self attention, all inputs are decoder inputs
        dec_output = self.self_masked_attention(
            zh_emb,
            zh_emb,
            zh_emb,
            self_mask,
            is_training)

        # query is decoder's outputs, key and value are encoder's inputs
        dec_output = self.context_attention(
            dec_output,
            en_emb,
            en_emb,
            mask,
            is_training)

        # decoder's output
        dec_output = self.feed_forward(dec_output)

        return dec_output

