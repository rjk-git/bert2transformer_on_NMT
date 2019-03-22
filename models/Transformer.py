from mxnet.gluon import nn
from models.Decoder import Decoder
from hyperParameters import GetHyperParameters as ghp


class Transformer(nn.Block):

    def __init__(self, zh_vocab_size, **kwargs):
        super(Transformer, self).__init__(**kwargs)

        self.decoder = Decoder(zh_vocab_size)
        with self.name_scope():
            # make the input english word embedding unit to the model_dim
            self.en_input_dense = nn.Dense(ghp.model_dims, flatten=False)
            # make the output pred dim to the size of chinese vocab_size
            self.linear = nn.Dense(zh_vocab_size, flatten=False)

    def forward(self, en_emb, en_idx, zh_idx):
        # en_emb shape : (batch_size, en_max_len, en_word_dim)
        # en_idx shape : (batch_size, en_max_len)
        # zh_idx shape : (batch_size, zh_max_len)

        # en_emb shape : (batch_size, en_max_len, model_dim)
        en_emb = self.en_input_dense(en_emb)

        output, dec_self_attn, context_attn = self.decoder(
            en_emb, en_idx, zh_idx)

        output = self.linear(output)
        return output, dec_self_attn, context_attn