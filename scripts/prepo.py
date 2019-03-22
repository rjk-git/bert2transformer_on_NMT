import numpy as np
import os
import re
import jieba
from collections import Counter
from hyperParameters import GetHyperParameters as ghp
from gluonnlp.data import BERTTokenizer, BERTSentenceTransform


def divide_data():
    lines = open("../data/neu_en_to_zh_dev.txt", "r", encoding="utf-8").readlines()
    en_list = []
    zh_list = []
    for line in lines:
        en_list.append(line.split("\t")[0])
        en_list.append("\n")
        zh_list.append(line.split("\t")[1].replace("\n", ""))
        zh_list.append("\n")
    open("../data/origin.en.sentences.txt", "w", encoding="utf-8").writelines(en_list)
    open("../data/origin.zh.sentences.txt", "w", encoding="utf-8").writelines(zh_list)


def load_en_vocab():
    vocab = [line.split("\t")[0] for line in open("vocab/en_vocab.tsv", "r", encoding="utf-8").readlines()
             if int(line.split("\t")[1].replace("\n", "")) >= ghp.min_count_vocab_size]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word


# ----------------------------------------------------------------------- #


def load_origin_sentences_data():
    en_origin = open("../data/origin.en.sentences.txt", "r", encoding="utf-8").readlines()
    zh_origin = open("../data/origin.zh.sentences.txt", "r", encoding="utf-8").readlines()

    en_sents = [sent.replace("\n", "") for sent in en_origin]
    zh_sents = [sent.replace("\n", "") for sent in zh_origin]

    return en_sents, zh_sents


def load_train_sentences_data():
    en_origin = open("../data/train.en.sentences.txt", "r", encoding="utf-8").readlines()
    zh_origin = open("../data/train.zh.sentences.txt", "r", encoding="utf-8").readlines()

    en_sents = [sent.replace("\n", "") for sent in en_origin]
    zh_sents = [sent.replace("\n", "") for sent in zh_origin]

    return en_sents, zh_sents


def save_train_sentences_data(en_sentences_data, zh_sentences_data):
    total_num = len(en_sentences_data)
    with open("../data/train.en.sentences.txt", "w", encoding="utf-8") as fw:
        for _ in range(total_num):
            fw.write(en_sentences_data[_])
            fw.write("\n")

    total_num = len(zh_sentences_data)
    with open("../data/train.zh.sentences.txt", "w", encoding="utf-8") as fw:
        for _ in range(total_num):
            fw.write(zh_sentences_data[_])
            fw.write("\n")


def make_vocab(file_path, file_name, vocab_size=None):
    text_lines = open(file_path, "r", encoding="utf-8").readlines()
    words = []
    for line in text_lines:
        line = line.rstrip().replace("\n", "")

        if file_name.startswith("zh"):
            line_words = [word for word in jieba.cut(line)]

        else:
            punctuation = """!,;:'@#$%%^&*()_+-=|~`<>.?"""
            re_punctuation = "[{}]+".format(punctuation)
            for p in re.findall(re_punctuation, line):
                line = line.replace(p, " {}".format(p))
            line_words = line.split()

        words.extend(line_words)
    print("总计：{}个词汇".format(len(words)))
    counter = Counter(words)
    if vocab_size is None:
        vocab_size = len(counter)
    if not os.path.exists("vocab"):
        os.makedirs("vocab")
    with open("vocab/{}.tsv".format(file_name), "w", encoding="utf-8") as fw:
        fw.write("{}\t100000\n{}\t100000\n{}\t100000\n{}\t100000\n".format("<pad>", "<unk>", "<bos>", "<eos>"))
        for word, count in counter.most_common(vocab_size):
            fw.write("{}\t{}\n".format(word, count))


def load_zh_vocab():
    vocab = [line.split("\t")[0] for line in open("vocab/zh_vocab.tsv", "r", encoding="utf-8").readlines()
             if int(line.split("\t")[1].replace("\n", "")) >= ghp.min_count_vocab_size]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word


def create_train_data(en_sentences, zh_sentences):
    zh2idx, idx2zh = load_zh_vocab()

    # word to idx
    en_list = []
    zh_list = []
    train_en_sentences = []
    train_zh_sentences = []
    punctuation = """!,;:'@#$%%^&*()_+=|~`<>.?"""
    re_punctuation = "[{}]+".format(punctuation)

    for en_sentence, zh_sentence in zip(en_sentences, zh_sentences):
        temp_sent = ""
        for p in re.findall(re_punctuation, en_sentence):
            temp_sent = en_sentence.replace(p, " {}".format(p))
        en = temp_sent.split()
        zh = [zh2idx.get(word, 1) for word in (" ".join([w for w in jieba.cut(zh_sentence)]) + " <eos>").split()]
        if len(en) <= ghp.max_seq_len-10 and len(zh) <= ghp.max_seq_len:
            en_list.append(en)
            zh_list.append(zh)
            train_en_sentences.append(en_sentence)
            train_zh_sentences.append(zh_sentence)

    # pad to max seq len
    ZH = np.zeros([len(zh_list), ghp.max_seq_len], np.int32)

    for i, zh in enumerate(zh_list):
        ZH[i] = np.lib.pad(zh, (0, ghp.max_seq_len - len(zh)), 'constant', constant_values=(0, 0))

    return train_en_sentences, train_zh_sentences, ZH


def get_en_sentences_loader(batch_size):

    if not os.path.exists("../data/train.en.sentences.txt") or not os.path.exists("../data/train.zh.sentences.txt"):
        origin_en_sentences, origin_zh_sentences = load_origin_sentences_data()
        train_en_sentences, train_zh_sentences, train_zh_idx = create_train_data(origin_en_sentences, origin_zh_sentences)
        save_train_sentences_data(train_en_sentences, train_zh_sentences)
        np.save("../data/train.zh.idx", train_zh_idx)
    else:
        train_zh_idx = np.load("../data/train.zh.idx.npy")
        train_en_sentences, train_zh_sentences = load_train_sentences_data()

    if len(train_en_sentences) != len(train_zh_idx):
        raise ValueError("train data is wrong! please make sure the chinese data and english data has the same"
                         " length.zh:{} vs en:{}".format(len(train_zh_idx), len(train_en_sentences)))

    for i in range(int(len(train_zh_idx) / batch_size) + 1):
        batch_en_sentences_data = train_en_sentences[i * batch_size: min(len(train_zh_idx), (i+1) * batch_size)]
        batch_zh_idx_data = train_zh_idx[i * batch_size: min(len(train_zh_idx), (i+1) * batch_size)]
        yield batch_en_sentences_data, batch_zh_idx_data


if __name__ == "__main__":
    # divide_data()
    # make_vocab("../data/origin.en.sentences.txt", "en_vocab", vocab_size=ghp.en_vocab_size)
    make_vocab("../data/origin.zh.sentences.txt", "zh_vocab", vocab_size=ghp.zh_vocab_size)
    # load_train_data()
    # loader = get_en_sentences_loader(64)
    # count1 = 0
    # count2 = 0
    # for en_sents, zh_idx in loader:
    #     print(len(en_sents))
    #     count1 += len(en_sents)
    #     print(zh_idx.shape)
    #     count2 += zh_idx.shape[0]
    # print(count1, count2)
    pass