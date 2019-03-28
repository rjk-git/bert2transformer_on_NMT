from mxnet import nd
from mxnet.gluon import nn
from models.decoder_layer import DecoderLayer
from models.positional_encoding import PositionalEncoding
from hyperParameters import GetHyperParameters as ghp


class Decoder(nn.Block):
    def __init__(self, zh_vocab_size, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        with self.name_scope():

            self.decoder_layers = []
            for i in range(ghp.layer_num):
                sub_layer = DecoderLayer()
                self.register_child(sub_layer)
                self.decoder_layers.append(sub_layer)

            self.seq_embedding = nn.Embedding(zh_vocab_size, ghp.model_dim)
            self.position_embedding = PositionalEncoding()

    def forward(self, en_emb, en_idx, zh_idx, is_training):
        output = self.seq_embedding(zh_idx)
        position = self.position_embedding(zh_idx)
        # zh_emb shape: (batch_size, seq_len, model_dim)
        zh_emb = output + position
        en_emb = self._padding_zero(en_emb, en_idx)
        dec_output = zh_emb
        for sub_layer in self.decoder_layers:
            dec_output = self._padding_zero(dec_output, zh_idx)
            dec_output = sub_layer(en_emb, dec_output, is_training)

        return dec_output

    def _padding_zero(self, emb_data, pad_idx):
        batch_size = emb_data.shape[0]
        pad_zero = nd.broadcast_not_equal(pad_idx, nd.array([[0]] * batch_size, ctx=ghp.ctx))

        pad_zero = nd.expand_dims(pad_zero, axis=2)

        pad_zero = nd.broadcast_axes(pad_zero, axis=2, size=ghp.model_dim)

        emb_data = emb_data * pad_zero
        return emb_data

