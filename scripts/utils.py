import os
import json
import pickle
import mxnet as mx
import numpy as np
from mxnet import nd
from gluonnlp import Vocab, data
from hyperParameters import GetHyperParameters as ghp
from typing import List

import gluonnlp
import mxnet as mx
from gluonnlp.data import BERTTokenizer, BERTSentenceTransform
from mxnet.gluon.data import DataLoader

from bert_embedding.dataset import BertEmbeddingDataset

def padding_mask(seq_k, seq_q):

    """
    eg:
        seq_k:
        [[31. 23. 12. 0.]
         [12. 1. 0. 0.]
         [15. 13. 0. 0.]
         [12. 11. 21. 0.]]
        <NDArray 4x4 @cpu(0)>

        seq_q:
        [[42. 3. 0. 0.]
         [123. 0. 0. 0.]
         [4. 45. 3. 0.]
         [5. 23. 12. 0.]]
        <NDArray 4x4 @cpu(0)>

        pad_mask:
        [[[1. 1. 1. 0.]
          [1. 1. 1. 0.]
          [0. 0. 0. 0.]
          [0. 0. 0. 0.]]

         [[1. 1. 0. 0.]
          [0. 0. 0. 0.]
          [0. 0. 0. 0.]
          [0. 0. 0. 0.]]

         [[1. 1. 0. 0.]
          [1. 1. 0. 0.]
          [1. 1. 0. 0.]
          [0. 0. 0. 0.]]

         [[1. 1. 1. 0.]
          [1. 1. 1. 0.]
          [1. 1. 1. 0.]
          [0. 0. 0. 0.]]]
        <NDArray 4x4x4 @cpu(0)>

        :param seq_k:
        :param seq_q:
        :return: pad_mask
    """
    _k = nd.not_equal(seq_k, 0)

    _q = nd.not_equal(seq_q, 0)

    k_e = nd.expand_dims(_k, axis=1)
    k_b = nd.broadcast_axes(k_e, axis=1, size=ghp.max_seq_len)

    q_e = nd.expand_dims(_q, axis=1)
    q_b = nd.broadcast_axes(q_e, axis=1, size=ghp.max_seq_len)
    q_t = nd.transpose(q_b, axes=(0, 2, 1))

    pad_mask = nd.greater(k_b + q_t, nd.array([1] * ghp.max_seq_len, ghp.ctx))
    return nd.transpose(pad_mask, axes=(0, 2, 1))


def sequence_mask(batch_seqs):
    batch_size, seq_len = batch_seqs.shape
    mask_matrix = np.ones(shape=(seq_len, seq_len), dtype=np.float)
    mask = np.triu(mask_matrix, k=0)
    mask = nd.expand_dims(nd.array(mask, ghp.ctx), axis=0)
    mask = nd.broadcast_axes(mask, axis=0, size=batch_size)
    return mask



def word_piece_tokenizer(sentences):
    ctx = mx.cpu()
    model = 'bert_12_768_12'
    dataset_name = 'book_corpus_wiki_en_uncased'
    max_seq_length = 20
    batch_size = 256
    _, vocab = gluonnlp.model.get_model(model,
                                        dataset_name=dataset_name,
                                        pretrained=True, ctx=ctx,
                                        use_pooler=False,
                                        use_decoder=False,
                                        use_classifier=False)
    tokenizer = BERTTokenizer(vocab)

    transform = BERTSentenceTransform(tokenizer=tokenizer,
                                      max_seq_length=max_seq_length,
                                      pair=False)
    dataset = BertEmbeddingDataset(sentences, transform)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    batches = []
    for token_ids, _, _ in data_loader:
        token_ids = token_ids.as_in_context(ctx)

        for token_id in token_ids.asnumpy():
            batches.append(token_id)

    cut_results = []
    for token_ids in batches:
        tokens = []
        for token_id in token_ids:
            if token_id == 1:
                break
            if token_id in (2, 3):
                continue
            token = vocab.idx_to_token[token_id]
            if token.startswith('##'):
                token = token[2:]
                tokens[-1] += token
            else:  # iv, avg last oov
                tokens.append(token)
        cut_results.append(tokens)
    return cut_results

if __name__ == '__main__':
    sentences = ["I want to see something.", "such as this picture is very beautiful,right?jackhoried.", "unaffable"]
    tokenizer = word_piece_tokenizer(sentences)
    print(tokenizer)
