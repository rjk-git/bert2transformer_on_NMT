import sys
sys.path.append("../")
import os
import multiprocessing
from mxnet.gluon import data
from mxnet import nd
from gluonnlp import Vocab
from gluonnlp.data import Counter
import json
import gluonnlp as nlp
import gluon
from gluonnlp.data import BERTSentenceTransform, BERTTokenizer
from pypinyin import pinyin
import re
BOS = "[unused16]"
EOS = "[unused17]"


class DatasetAssiantTransformer():
    def __init__(self, en_vocab=None, ch_vocab=None, max_en_len=None, max_ch_len=None):
        self.en_vocab = en_vocab
        self.ch_vocab = ch_vocab
        self.max_en_len = max_en_len
        self.max_ch_len = max_ch_len
        self.bert_en_tokenzier = BERTTokenizer(en_vocab)
        self.bert_ch_tokenzier = BERTTokenizer(ch_vocab)

    def MTSentenceProcess(self, *transentence_and_aimsentence):
        transentence, aimsentence = transentence_and_aimsentence
        assert isinstance(transentence, str), 'the input type must be str'
        assert isinstance(aimsentence, str), 'the input type must be str'

        transentence = self.bert_en_tokenzier(transentence)
        aimsentence = self.bert_ch_tokenzier(aimsentence)

        transentence = [self.en_vocab.cls_token] + \
            transentence + [self.en_vocab.sep_token]
        aimsentence = [BOS] + \
            aimsentence + [self.ch_vocab.sep_token]

        if self.max_en_len and len(transentence) > self.max_en_len:
            transentence = transentence[0:self.max_en_len]
        if self.max_ch_len and len(aimsentence) > self.max_ch_len:
            aimsentence = aimsentence[0:self.max_ch_len]

        trans_valid_len = len(transentence)
        aim_valid_len = len(aimsentence)

        transentence = self.en_vocab[transentence]
        aimsentence = self.ch_vocab[aimsentence]
        label = aimsentence[1:] + [self.ch_vocab(EOS)]

        return transentence, aimsentence, label, trans_valid_len, aim_valid_len


class MTDataLoader(object):
    def __init__(self, dataset, batch_size, assiant, shuffle=False, num_workers=3, lazy=True):
        trans_func = assiant.MTSentenceProcess
        self.dataset = dataset.transform(trans_func, lazy=lazy)
        self.batch_size = batch_size
        self.en_pad_val = assiant.en_vocab[assiant.en_vocab.padding_token]
        self.ch_pad_val = assiant.ch_vocab[assiant.ch_vocab.padding_token]
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.dataloader = self._build_dataloader()

    def _build_dataloader(self):
        batchify_fn = nlp.data.batchify.Tuple(
            nlp.data.batchify.Pad(pad_val=self.en_pad_val),
            nlp.data.batchify.Pad(pad_val=self.ch_pad_val),
            nlp.data.batchify.Pad(pad_val=self.ch_pad_val),
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
