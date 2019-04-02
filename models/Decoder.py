import math
from mxnet import nd
from mxnet.gluon import nn
from models.decoder_layer import DecoderLayer
from hyperParameters import GetHyperParameters as ghp


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
        en_emb = self.padding_with_zero(en_emb, en_idx)
        dec_output = zh_emb

        for sub_layer in self.decoder_layers:
            dec_output = self.padding_with_zero(dec_output, zh_idx)
            dec_output = sub_layer(en_emb, dec_output, is_training)

        return dec_output

    @staticmethod
    def padding_with_zero(emb_data, pad_idx):
        batch_size = emb_data.shape[0]

        pad_zero = nd.broadcast_not_equal(pad_idx, nd.array([[0]] * batch_size, ctx=ghp.ctx))

        pad_zero = nd.expand_dims(pad_zero, axis=2)

        pad_zero = nd.broadcast_axes(pad_zero, axis=2, size=ghp.model_dim)

        emb_data = emb_data * pad_zero

        return emb_data

    @staticmethod
    def _get_position_encoding(length, min_timescale=1.0, max_timescale=1.0e4):
        """Return positional encoding.
        Args:
          length: Sequence length.
          hidden_size: Size of the model_dim
          min_timescale: Minimum scale that will be applied at each position
          max_timescale: Maximum scale that will be applied at each position

        Returns:
          Tensor with shape [length, hidden_size]
        """
        position = nd.arange(length, ctx=ghp.ctx)
        num_timescales = ghp.model_dim // 2
        log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
        inv_timescales = min_timescale * nd.exp(nd.arange(num_timescales, ctx=ghp.ctx) * -log_timescale_increment)
        scaled_time = nd.expand_dims(position, 1) * nd.expand_dims(inv_timescales, 0)
        signal = nd.concat(nd.sin(scaled_time), nd.cos(scaled_time), dim=1)
        return signal