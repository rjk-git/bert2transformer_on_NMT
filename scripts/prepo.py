import numpy as np
import os
import re
import jieba
from collections import Counter
from hyperParameters import GetHyperParameters as ghp
from scripts.utils import word_piece_tokenizer
from tqdm import tqdm
from mxnet import nd


def divide_data():
    first_file = open("./temp_data/" + os.listdir("./temp_data/")[0], "r", encoding="utf-8")
    lines = first_file.readlines()
    en_list = []
    zh_list = []
    for line in lines:
        en_list.append(line.split("\t")[0])
        en_list.append("\n")
        zh_list.append(line.split("\t")[1].replace("\n", ""))
        zh_list.append("\n")
    open(ghp.origin_en_train_file, "w", encoding="utf-8").writelines(en_list)
    open(ghp.origin_zh_train_file, "w", encoding="utf-8").writelines(zh_list)


def load_en_vocab():
    vocab = [line.split("\t")[0] for line in open(ghp.en_vocab_file, "r", encoding="utf-8").readlines()
             if int(line.split("\t")[1].replace("\n", "")) >= ghp.min_count_vocab_size]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word


# ----------------------------------------------------------------------- #


def load_origin_sentences_data():
    en_origin = open(ghp.origin_en_train_file, "r", encoding="utf-8").readlines()
    zh_origin = open(ghp.origin_zh_train_file, "r", encoding="utf-8").readlines()

    en_sentences = [sent.replace("\n", "") for sent in en_origin]
    zh_sentences = [sent.replace("\n", "") for sent in zh_origin]

    return en_sentences, zh_sentences


def load_train_sentences_data():
    en_origin = open(ghp.handled_en_train_file, "r", encoding="utf-8").readlines()
    zh_origin = open(ghp.handled_zh_train_file, "r", encoding="utf-8").readlines()

    en_sentences = [sent.replace("\n", "") for sent in en_origin]
    zh_sentences = [sent.replace("\n", "") for sent in zh_origin]

    return en_sentences, zh_sentences


def save_train_sentences_data(en_sentences_data, zh_sentences_data):
    total_num = len(en_sentences_data)
    with open(ghp.handled_en_train_file, "w", encoding="utf-8") as fw:
        for _ in range(total_num):
            fw.write(en_sentences_data[_])
            fw.write("\n")

    total_num = len(zh_sentences_data)
    with open(ghp.handled_zh_train_file, "w", encoding="utf-8") as fw:
        for _ in range(total_num):
            fw.write(zh_sentences_data[_])
            fw.write("\n")


def make_zh_vocab(file_path, vocab_size=None):
    text_lines = open(file_path, "r", encoding="utf-8").readlines()
    words = []
    for line in text_lines:
        line = line.rstrip().replace("\n", "")
        line_words = [word for word in jieba.cut(line)]
        words.extend(line_words)
    print("获取：{}个词汇".format(len(words)))
    if vocab_size is not None:
        print("词典：{}个词汇".format(vocab_size))
    counter = Counter(words)
    if vocab_size is None:
        vocab_size = len(counter)
    if not os.path.exists("vocab"):
        os.makedirs("vocab")
    with open(ghp.zh_vocab_file, "w", encoding="utf-8") as fw:
        fw.write("{}\t100000\n{}\t100000\n{}\t100000\n{}\t100000\n".format("<pad>", "<unk>", "<bos>", "<eos>"))
        for word, count in counter.most_common(vocab_size):
            fw.write("{}\t{}\n".format(word, count))


def make_zh_vocab2(file_path, vocab_size=None):
    text_lines = open(file_path, "r", encoding="utf-8").readlines()
    words = []
    for line in text_lines:
        line = line.rstrip().replace("\n", "")
        line_words = [word for word in line.split()]
        words.extend(line_words)
    print("获取：{}个词汇".format(len(words)))
    if vocab_size is not None:
        print("词典：{}个词汇".format(vocab_size))
    counter = Counter(words)
    if vocab_size is None:
        vocab_size = len(counter)
    if not os.path.exists("vocab"):
        os.makedirs("vocab")
    with open(ghp.zh_vocab_file, "w", encoding="utf-8") as fw:
        fw.write("{}\t100000\n{}\t100000\n{}\t100000\n{}\t100000\n".format("<pad>", "<unk>", "<bos>", "<eos>"))
        for word, count in counter.most_common(vocab_size):
            fw.write("{}\t{}\n".format(word, count))


def load_zh_vocab():
    vocab = [line.split("\t")[0] for line in open(ghp.zh_vocab_file, "r", encoding="utf-8").readlines()
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
    ens = word_piece_tokenizer(en_sentences)
    for i, (en_sentence, zh_sentence) in enumerate(zip(en_sentences, zh_sentences)):
        zh = [zh2idx.get(word, 1) for word in (" ".join([w for w in jieba.cut(zh_sentence)]) + " <eos>").split()]
        if len(ens[i]) <= ghp.max_seq_len and len(zh) <= ghp.max_seq_len:
            en_list.append(ens[i])
            zh_list.append(zh)
            train_en_sentences.append(en_sentence)
            train_zh_sentences.append(zh_sentence)

    # pad to max seq len
    train_zh_idx = np.zeros([len(zh_list), ghp.max_seq_len], np.int32)

    for i, zh in enumerate(zh_list):
        train_zh_idx[i] = np.lib.pad(zh, (0, ghp.max_seq_len - len(zh)), 'constant', constant_values=(0, 0))

    return train_en_sentences, train_zh_sentences, train_zh_idx


def create_train_data2(zh_sentences):
    zh2idx, idx2zh = load_zh_vocab()

    # word to idx
    zh_list = []

    with tqdm(total=len(zh_sentences)) as bar:
        for zh_sentence in zh_sentences:
            bar.update(1)
            zh = [zh2idx.get(word, 1) for word in (zh_sentence + " <eos>").split()]
            zh_list.append(zh)

    # pad to max seq len
    train_zh_idx = np.zeros([len(zh_list), ghp.max_seq_len], np.int32)

    with tqdm(total=len(zh_list)) as bar:
        for i, zh in enumerate(zh_list):
            bar.update(1)
            train_zh_idx[i] = np.lib.pad(zh, (0, ghp.max_seq_len - len(zh)), 'constant', constant_values=(0, 0))

    return train_zh_idx


def get_data_loader():
    batch_size = ghp.batch_size

    # process sentences
    if not os.path.exists(ghp.handled_en_train_file) or not os.path.exists(ghp.handled_zh_train_file):
        origin_en_sentences, origin_zh_sentences = load_origin_sentences_data()
        train_en_sentences, train_zh_sentences, train_zh_idx = create_train_data(origin_en_sentences, origin_zh_sentences)
        save_train_sentences_data(train_en_sentences, train_zh_sentences)
        np.save(ghp.handled_zh_idx_file_name, train_zh_idx)
    # get processed sentences and idx zh
    else:
        train_zh_idx = np.load(ghp.handled_zh_idx_file_name + ".npy")
        train_en_sentences, train_zh_sentences = load_train_sentences_data()

    # check the en and zh has same length
    if len(train_en_sentences) != len(train_zh_idx):
        raise ValueError("train data is wrong! please make sure the chinese data and english data has the same"
                         " length.zh:{} vs en:{}".format(len(train_zh_idx), len(train_en_sentences)))

    # make an iterator
    for i in range(int(len(train_zh_idx) / batch_size) + 1):
        batch_en_sentences_data = train_en_sentences[i * batch_size: min(len(train_zh_idx), (i+1) * batch_size)]
        batch_zh_idx_data = train_zh_idx[i * batch_size: min(len(train_zh_idx), (i+1) * batch_size)]
        yield batch_en_sentences_data, batch_zh_idx_data


def get_data_loader2():
    print("#######开始加载训练数据：英文， 中文（已分词）#########")
    batch_size = ghp.batch_size

    origin_en_sentences, origin_zh_sentences = load_origin_sentences_data()
    print("获取训练数据：中文{}条句子，英文{}条句子成功！".format(len(origin_zh_sentences), len(origin_en_sentences)))

    if os.path.exists(ghp.handled_zh_idx_file_name + ".npy"):
        train_zh_idx = np.load(ghp.handled_zh_idx_file_name + ".npy")
    else:
        train_zh_idx = create_train_data2(origin_zh_sentences)
        np.save(ghp.handled_zh_idx_file_name, train_zh_idx)
    print("生成训练数据：中文句子索引{}条成功！".format(len(train_zh_idx)))

    # check the en and zh has same length
    if len(origin_en_sentences) != len(train_zh_idx):
        raise ValueError("train data is wrong! please make sure the chinese data and english data has the same"
                         " length.zh:{} vs en:{}".format(len(train_zh_idx), len(origin_en_sentences)))

    # make an iterator
    for i in range(int(len(train_zh_idx) / batch_size) + 1):
        batch_en_sentences_data = origin_en_sentences[i * batch_size: min(len(train_zh_idx), (i+1) * batch_size)]
        batch_zh_idx_data = train_zh_idx[i * batch_size: min(len(train_zh_idx), (i+1) * batch_size)]
        yield batch_en_sentences_data, batch_zh_idx_data


if __name__ == "__main__":
    # get vocab
    # make_zh_vocab2(ghp.origin_zh_train_file, vocab_size=ghp.zh_vocab_size)
    # make_zh_vocab(ghp.origin_zh_train_file, vocab_size=ghp.zh_vocab_size)
    divide_data()