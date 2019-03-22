import numpy as np
from mxnet import nd
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
from bert_embedding import BertEmbedding
from gluonnlp.data import BERTTokenizer, BERTSentenceTransform

tokenizer = BERTTokenizer()
list = tokenizer.basic_tokenizer._tokenize("this is a test.")
print(list)