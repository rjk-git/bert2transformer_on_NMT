# import numpy as np
# from mxnet import nd
# from bert_embedding import BertEmbedding
# from prepo import load_en_vocab
# from mxnet.gluon import data as gdata
# _, idx2en = load_en_vocab()
#
# lines = open("../data/not.origin.en.sentences.txt", "r", encoding="utf-8").readlines()
# list = []
# for line in lines[:10]:
#     list.append(line.replace("\n", ""))
#
# print(len(list))
#
# sentences = list
# bert = BertEmbedding()
# result = bert(sentences)
#
# print(len(result))
# print(result[0][0])
# print(result[0][1][0].tolist())
import os
from mxnet import nd
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss as sceloss

label = nd.array([[15, 15, 15, 15, 15],
                  [3, 5, 2, 1, 1],
                  [3, 5, 2, 1, 1],
                  [3, 5, 2, 1, 1],
                  [3, 5, 2, 1, 1]])

pred = nd.array([[3, 1, 1, 2, 2],
                  [3, 5, 2, 1, 1],
                  [3, 5, 2, 1, 1],
                  [3, 5, 2, 1, 1],
                  [3, 5, 2, 1, 1]])

loss = sceloss(axis=-1, sparse_label=False, from_logits=False)
print(loss(pred, label))