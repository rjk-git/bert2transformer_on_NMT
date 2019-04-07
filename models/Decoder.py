import math

from mxnet import nd
from mxnet.gluon import nn
from models.decoder_layer import DecoderLayer

from hyperParameters import GetHyperParameters as ghp
from models.utils import getMask, getSelfMask


class Decoder(nn.Block):
    def __init__(self, ch_vocab_size, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        with self.name_scope():

            self.decoder_layers = []
            for i in range(ghp.layer_num):
                sub_layer = DecoderLayer()
                self.register_child(sub_layer)
                self.decoder_layers.append(sub_layer)

            self.seq_embedding = nn.Embedding(ch_vocab_size, ghp.model_dim)

    def forward(self, en_emb, en_idx, zh_idx, is_training):
        output = self.seq_embedding(zh_idx)

        position = self._get_position_encoding(len(zh_idx[0]))

        # zh_emb shape: (batch_size, seq_len, model_dim)
        zh_emb = output + position

        # replace the pad with 0
        mask = getMask(zh_idx, en_idx)
        self_mask = getSelfMask(zh_idx)
        dec_output = zh_emb

        for sub_layer in self.decoder_layers:
            dec_output = sub_layer(en_emb, dec_output, mask, self_mask, is_training)
        return dec_output

    @staticmethod
    def _get_position_encoding(length, min_timescale=1.0, max_timescale=1.0e4):
        position = nd.arange(length, ctx=ghp.ctx)
        num_timescales = ghp.model_dim // 2
        log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
        inv_timescales = min_timescale * nd.exp(nd.arange(num_timescales, ctx=ghp.ctx) * -log_timescale_increment)
        scaled_time = nd.expand_dims(position, 1) * nd.expand_dims(inv_timescales, 0)
        signal = nd.concat(nd.sin(scaled_time), nd.cos(scaled_time), dim=1)
        return signal
