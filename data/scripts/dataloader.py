import json
import multiprocessing
import os
import re
import sys
sys.path.append("../")

import gluonnlp as nlp
from gluonnlp import Vocab
from gluonnlp.data import BERTSentenceTransform, BERTTokenizer
from mxnet import nd
from mxnet.gluon import data


class DatasetAssiantTransformer():
    def __init__(self, src_vocab=None, tgt_vocab=None, max_src_len=None, max_tgt_len=None):
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.bert_src_tokenzier = BERTTokenizer(src_vocab)
        self.bert_tgt_tokenzier = BERTTokenizer(tgt_vocab)
        self.bos_token = "[CLS]"
        self.eos_token = "[SEP]"

    def MTSentenceProcess(self, *src_and_tgt):
        src, tgt = src_and_tgt
        assert isinstance(src, str), 'the input type must be str'
        assert isinstance(tgt, str), 'the input type must be str'

        src = self.bert_src_tokenzier(src)
        tgt = self.bert_tgt_tokenzier(tgt)

        if self.max_src_len and len(src) > self.max_src_len-2:
            src = src[0:self.max_src_len-2]
        if self.max_tgt_len and len(tgt) > self.max_tgt_len-1:
            tgt = tgt[0:self.max_tgt_len-1]

        src = [self.src_vocab.cls_token] + \
            src + [self.src_vocab.sep_token]
        tgt = [self.bos_token] + tgt

        src_valid_len = len(src)
        tgt_valid_len = len(tgt)

        src = self.src_vocab[src]
        tgt = self.tgt_vocab[tgt]
        label = tgt[1:] + [self.tgt_vocab(self.eos_token)]

        return src, tgt, label, src_valid_len, tgt_valid_len


class MTDataLoader(object):
    def __init__(self, dataset, batch_size, assiant, shuffle=False, num_workers=3, lazy=True):
        trans_func = assiant.MTSentenceProcess
        self.dataset = dataset.transform(trans_func, lazy=lazy)
        self.batch_size = batch_size
        self.src_pad_val = assiant.src_vocab[assiant.src_vocab.padding_token]
        self.tgt_pad_val = assiant.tgt_vocab[assiant.tgt_vocab.padding_token]
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.dataloader = self._build_dataloader()

    def _build_dataloader(self):
        batchify_fn = nlp.data.batchify.Tuple(
            nlp.data.batchify.Pad(pad_val=self.src_pad_val),
            nlp.data.batchify.Pad(pad_val=self.tgt_pad_val),
            nlp.data.batchify.Pad(pad_val=self.tgt_pad_val),
            nlp.data.batchify.Stack(dtype="float32"),
            nlp.data.batchify.Stack(dtype="float32"),)
        dataloader = data.DataLoader(dataset=self.dataset, batch_size=self.batch_size,
                                     shuffle=self.shuffle, batchify_fn=batchify_fn,
                                     num_workers=self.num_workers)
        return dataloader

    @property
    def dataiter(self):
        return self.dataloader

    @property
    def data_lengths(self):
        return len(self.dataset)
