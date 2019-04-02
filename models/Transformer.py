from mxnet.gluon import nn
from models.Decoder import Decoder
from hyperParameters import GetHyperParameters as ghp


class Transformer(nn.Block):

    def __init__(self, ch_vocab_size, **kwargs):
        super(Transformer, self).__init__(**kwargs)

        # here no need Encoder because the pre-trained word-embedding by bert has done this work
        self.decoder = Decoder(ch_vocab_size)
        with self.name_scope():
            # make the input english word embedding unit to the model_dim
            self.en_input_dense = nn.Dense(ghp.model_dim, use_bias=False, flatten=False)
            # make the output pred dim to the size of chinese vocab_size
            self.linear = nn.Dense(ch_vocab_size, use_bias=False, flatten=False)

    def forward(self, en_emb, en_idx, zh_idx, is_training):
        # en_emb shape : (batch_size, en_max_len, en_word_dim)
        # en_idx shape : (batch_size, en_max_len)
        # zh_idx shape : (batch_size, zh_max_len)

        # en_emb shape : (batch_size, en_max_len, model_dim)
        en_emb = self.en_input_dense(en_emb)

        output = self.decoder(
            en_emb, en_idx, zh_idx, is_training)

        # output shape : (batch_size, max_seq_len, ch_vocab_size)
        output = self.linear(output)
        return output
