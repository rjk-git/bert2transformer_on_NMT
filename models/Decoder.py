from mxnet import nd
from mxnet.gluon import nn
from models.decoder_layer import DecoderLayer
from models.positional_encoding import PositionalEncoding
from utils import padding_mask, sequence_mask
from hyperParameters import GetHyperParameters as ghp


class Decoder(nn.Block):
    def __init__(self, zh_vocab_size, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        with self.name_scope():

            self.decoder_layers = []
            for i in range(ghp.layer_nums):
                sub_layer = DecoderLayer()
                self.register_child(sub_layer)
                self.decoder_layers.append(sub_layer)

            self.seq_embedding = nn.Embedding(zh_vocab_size + 1, ghp.model_dims)
            self.position_embedding = PositionalEncoding()

    def forward(self, en_emb, en_idx, zh_idx):
        output = self.seq_embedding(zh_idx)
        position = self.position_embedding(zh_idx)
        zh_emb = output + position

        # make a self masked matrix, put 0 in where word is pad and where word in the down triangle
        #  pad_mask             seq_mask                self_attn_mask
        # [1,1,1,0]            [1,1,1,1]                 [1,1,1,0]
        # [1,1,1,0]     +      [0,1,1,1]       =         [0,1,1,0]
        # [1,1,1,0]            [0,0,1,1]                 [0,0,1,0]
        # [1,1,1,0]            [0,0,0,1]                 [0,0,0,0]
        self_attn_pad_mask = padding_mask(zh_idx, zh_idx)
        self_seq_mask = sequence_mask(zh_idx)
        zeros = nd.ones((zh_idx.shape[1], 1), ctx=zh_idx.context)
        self_attn_mask = nd.broadcast_greater(self_attn_pad_mask+self_seq_mask, zeros)

        # make a context masked matrix, put 0 in where word is pad
        context_attn_mask = padding_mask(en_idx, zh_idx)
        self_attentions = []
        context_attentions = []
        output = zh_emb

        for sub_layer in self.decoder_layers:
            output, self_attn, context_attn = sub_layer(en_emb, output, self_attn_mask, context_attn_mask)
            self_attentions.append(self_attn)
            context_attentions.append(context_attn)

        return output, self_attentions, context_attentions

