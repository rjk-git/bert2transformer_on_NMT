import math
import gluonnlp
import numpy as np

from mxnet import nd
from mxnet.gluon import nn
from bert_embedding.dataset import BertEmbeddingDataset
from gluonnlp.data import BERTTokenizer, BERTSentenceTransform
from mxnet.gluon.data import DataLoader

from scripts.hyperParameters import GetHyperParameters as ghp


class Transformer(nn.Block):

    def __init__(self, ch_vocab_size, **kwargs):
        super(Transformer, self).__init__(**kwargs)

        self.encoder = Encoder()
        self.decoder = Decoder(ch_vocab_size)

        with self.name_scope():

            self.en_output_dense = nn.Dense(
                ghp.model_dim, use_bias=False, flatten=False, dtype=ghp.dtype)

            self.linear = nn.Dense(
                ch_vocab_size, use_bias=False, flatten=False, dtype=ghp.dtype)

    def forward(self, sentences_left, idx_right, is_training):
        # en_emb shape : (batch_size, en_max_len, en_word_dim)
        # en_idx shape : (batch_size, en_max_len)
        # zh_idx shape : (batch_size, zh_max_len)
        emb_left, idx_left = self.encoder(sentences_left)

        # en_emb shape : (batch_size, en_max_len, model_dim)
        emb_left = self.en_output_dense(emb_left)
        output = self.decoder(
            emb_left, idx_left, idx_right, is_training)

        # output shape : (batch_size, max_seq_len, ch_vocab_size)
        output = self.linear(output)

        # print(output)
        return output


class Encoder(nn.Block):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.ctx = ghp.ctx
        self.max_seq_length = ghp.max_seq_len
        self.batch_size = ghp.batch_size
        self.bert, self.vocab = gluonnlp.model.get_model('bert_12_768_12',
                                                         dataset_name='book_corpus_wiki_en_uncased',
                                                         pretrained=True, ctx=self.ctx,
                                                         use_pooler=False,
                                                         use_decoder=False,
                                                         use_classifier=False)

    def forward(self, sentences, *args):
        tokenizer = BERTTokenizer(self.vocab)
        transform = BERTSentenceTransform(tokenizer=tokenizer,
                                          max_seq_length=self.max_seq_length,
                                          pair=False)
        data_set = BertEmbeddingDataset(sentences, transform)
        token_ids = []
        valid_lengths = []
        token_types = []
        for token_id, valid_length, token_type in data_set:
            token_ids.append(token_id.tolist())
            valid_lengths.append(valid_length.tolist())
            token_types.append(token_type.tolist())

        token_ids = nd.array(token_ids, ctx=ghp.ctx)
        valid_lengths = nd.array(valid_lengths, ctx=ghp.ctx)
        token_types = nd.array(token_types, ctx=ghp.ctx)

        sequence_outputs = self.bert(token_ids, token_types, valid_lengths)

        zeros = nd.zeros_like(token_ids)
        token_ids = nd.where(nd.equal(token_ids, 1), zeros, token_ids)

        return sequence_outputs, token_ids


class Decoder(nn.Block):
    def __init__(self, ch_vocab_size, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        with self.name_scope():

            self.decoder_layers = []
            for i in range(ghp.layer_num):
                sub_layer = DecoderLayer()
                self.register_child(sub_layer)
                self.decoder_layers.append(sub_layer)

            self.seq_embedding = nn.Embedding(
                ch_vocab_size, ghp.model_dim, dtype=ghp.dtype)

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
            dec_output = sub_layer(
                en_emb, dec_output, mask, self_mask, is_training)
        return dec_output

    @staticmethod
    def _get_position_encoding(length, min_timescale=1.0, max_timescale=1.0e4):
        position = nd.arange(length, ctx=ghp.ctx, dtype=ghp.dtype)
        num_timescales = ghp.model_dim // 2
        log_timescale_increment = (math.log(
            float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
        inv_timescales = min_timescale * \
            nd.exp(nd.arange(num_timescales, ctx=ghp.ctx,
                             dtype=ghp.dtype) * -log_timescale_increment)
        scaled_time = nd.expand_dims(
            position, 1) * nd.expand_dims(inv_timescales, 0)
        signal = nd.concat(nd.sin(scaled_time), nd.cos(scaled_time), dim=1)
        return signal


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


class MultiHeadAttention(nn.Block):
    def __init__(self, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        with self.name_scope():
            self.queries_dense = nn.Dense(
                ghp.model_dim, use_bias=False, flatten=False)
            self.keys_dense = nn.Dense(
                ghp.model_dim, use_bias=False, flatten=False)
            self.values_dense = nn.Dense(
                ghp.model_dim, use_bias=False, flatten=False)
            self.dropout = nn.Dropout(ghp.dropout)
            self.LayerNorm = nn.LayerNorm(epsilon=ghp.norm_epsilon)

    def forward(self, queries, keys, values, mask, is_training=True):
        # queries shape: (batch_size, q_len, model_dim)
        # keys shape: (batch_size, k_len, model_dim)
        # values shape: (batch_size, k_len, model_dim)

        # residual
        residual = queries
        batch_size = queries.shape[0]
        q_len = queries.shape[1]
        k_len = keys.shape[1]

        Q = self.queries_dense(queries)
        K = self.keys_dense(keys)
        V = self.values_dense(values)

        c_dim = int(ghp.model_dim / ghp.head_num)

        # Qs(list) shape (batch_size, q_len, c_dim) * head_num
        # Ks(list) shape (batch_size, k_len, c_dim) * head_num
        # Vs(list) shape (batch_size, k_len, c_dim) * head_num
        Qs = nd.split(Q, num_outputs=ghp.head_num, axis=-1)
        Ks = nd.split(K, num_outputs=ghp.head_num, axis=-1)
        Vs = nd.split(V, num_outputs=ghp.head_num, axis=-1)

        # Q_ shape (batch_size * num_head, q_len, c_dim)
        Q_ = nd.empty(shape=(1, q_len, c_dim), ctx=ghp.ctx)
        for q in Qs:
            Q_ = nd.concat(Q_, q, dim=0)
        Q_ = Q_[1:]

        # K_ shape (batch_size * num_head, k_len, c_dim)
        K_ = nd.empty(shape=(1, k_len, c_dim), ctx=ghp.ctx)
        for k in Ks:
            K_ = nd.concat(K_, k, dim=0)
        K_ = K_[1:]

        # V_ shape (batch_size * num_head, k_len, c_dim)
        V_ = nd.empty(shape=(1, k_len, c_dim), ctx=ghp.ctx)
        for v in Vs:
            V_ = nd.concat(V_, v, dim=0)
        V_ = V_[1:]

        # batch01: (seq01_len, c_dim01)
        # batch02: (seq02_len, c_dim01)
        # batch03: (seq03_len, c_dim01)
        # ...
        # batch09: (seq01_len, c_dim02)
        # ...

        scale = ghp.c_dim ** -0.5

        # att_score
        # shape: (batch_size * head_num, q_len, k_len)
        att_scores = nd.batch_dot(Q_, K_, transpose_b=True)

        # scale
        # shape: (batch_size * head_num, q_len, k_len)
        att_scores = att_scores * scale

        # mask
        mask = nd.expand_dims(mask, axis=0)
        mask = nd.broadcast_axes(mask, axis=0, size=ghp.head_num)
        mask = nd.reshape(mask, shape=(-1, q_len, k_len))
        padding = nd.ones_like(mask) * -np.inf
        att_scores = nd.where(nd.equal(mask, 0), padding, att_scores)

        # attention in window
        win_mask = getWindowAttentionMask(q_len, k_len)
        win_mask = nd.expand_dims(win_mask, axis=0)
        win_mask = nd.broadcast_axes(win_mask, axis=0, size=ghp.head_num * ghp.batch_size)
        padding = nd.ones_like(win_mask) * -np.inf
        att_scores = nd.where(nd.equal(win_mask, 0), padding, att_scores)

        # att_weights shape: (batch_size * head_num, q_len, k_len)
        att_weights = nd.softmax(att_scores, axis=-1)
        output = nd.batch_dot(att_weights, V_)

        outputs = nd.split(output, num_outputs=ghp.head_num, axis=0)
        empty = nd.empty(shape=(batch_size, q_len, 1), ctx=ghp.ctx)
        for out in outputs:
            empty = nd.concat(empty, out, dim=-1)
        output = empty[:, :, 1:]

        if is_training:
            output = self.dropout(output)

        output = output + residual
        output = self.LayerNorm(output)
        return output


class FeedForward(nn.Block):
    def __init__(self, **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        with self.name_scope():
            self.ffn_dense = nn.Dense(
                ghp.ffn_dim, activation="relu", use_bias=True, flatten=False, dtype=ghp.dtype)
            self.model_dense = nn.Dense(
                ghp.model_dim, use_bias=True, flatten=False, dtype=ghp.dtype)
            self.dropout = nn.Dropout(ghp.ffn_dropout)
            self.layer_norm = nn.LayerNorm(axis=-1, epsilon=ghp.norm_epsilon)

    def forward(self, x, *args):
        # x shape : (batch_size, seq_len, model_dim)
        residual = x

        # output shape : (batch_size, seq_len, ffn_dim)
        output = self.ffn_dense(x)

        # output shape : (batch_size, seq_len, model_dim)
        output = self.model_dense(output)

        # shape : (batch_size, seq_len, model_dim)
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output


def getMask(q_seq, k_seq):
    # q_seq shape : (batch_size, q_seq_len)
    # k_seq shape : (batch_size, k_seq_len)
    q_len = q_seq.shape[1]
    pad_mask = nd.not_equal(k_seq, 0)
    pad_mask = nd.expand_dims(pad_mask, axis=1)
    pad_mask = nd.broadcast_axes(pad_mask, axis=1, size=q_len)

    return pad_mask


def getSelfMask(q_seq):

    batch_size, seq_len = q_seq.shape
    mask_matrix = np.ones(shape=(seq_len, seq_len), dtype=ghp.dtype)

    mask = np.tril(mask_matrix, k=0)

    mask = nd.expand_dims(nd.array(mask, ctx=ghp.ctx, dtype=ghp.dtype), axis=0)

    mask = nd.broadcast_axes(mask, axis=0, size=batch_size)

    return mask


def getWindowAttentionMask(q_len, k_len, window_size=15):
    e = nd.eye(q_len, k_len, 0, ctx=ghp.ctx)
    for k in range(1, window_size + 1):
        e0 = nd.eye(q_len, k_len, k, ctx=ghp.ctx)
        e1 = nd.eye(q_len, k_len, -k, ctx=ghp.ctx)
        e = e + e0 + e1
    return e


if __name__ == '__main__':
    getWindowAttentionMask(8, 10)
