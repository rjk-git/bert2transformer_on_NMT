import os
import jieba
import numpy as np
import multiprocessing

from tqdm import tqdm
from collections import Counter

from hyperParameters import GetHyperParameters as ghp

ch2idx = {}
need_cut_sentences_num = 0


def load_origin_sentences_data():
    en_origin = open(ghp.origin_en_train_file, "r",
                     encoding="utf-8").readlines()
    zh_origin = open(ghp.origin_ch_train_file, "r",
                     encoding="utf-8").readlines()

    # randnum = random.randint(0, 100)

    en_sentences = [sent.replace("\n", "") for sent in en_origin]
    # random.seed(randnum)
    # random.shuffle(en_sentences)

    zh_sentences = [sent.replace("\n", "") for sent in zh_origin]
    # random.seed(randnum)
    # random.shuffle(zh_sentences)

    return en_sentences, zh_sentences


def load_train_sentences_data():
    en_origin = open(ghp.handled_en_train_file, "r",
                     encoding="utf-8").readlines()
    zh_origin = open(ghp.handled_zh_train_file, "r",
                     encoding="utf-8").readlines()

    en_sentences = [sent.replace("\n", "") for sent in en_origin]
    zh_sentences = [sent.replace("\n", "") for sent in zh_origin]

    sents_left = []
    sents_right = []
    max_len = 0
    for sentence_left, sentence_right in zip(en_sentences, zh_sentences):
        if len(sentence_left) > ghp.max_seq_len or len(
                sentence_right.split()) > ghp.max_seq_len - 1:
            continue
        if len(sentence_right.split()) < 4:
            continue
        if len(sentence_left) > max_len:
            max_len = len(sentence_left)
        sents_left.append(sentence_left)
        sents_right.append(sentence_right)
    print("max_len:{}".format(max_len))

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


def make_ch_vocab():
    file_path = ghp.origin_ch_train_file
    text_lines = open(file_path, "r", encoding="utf-8").readlines()
    words = []
    for line in text_lines:
        line = line.rstrip().replace("\n", "")
        line_words = [word for word in line.split()]
        words.extend(line_words)
    print("获取：{}个词汇".format(len(words)))

    vocab_words = []

    counter = Counter(words)
    vocab_size = ghp.ch_vocab_size
    for word, _ in counter.most_common(vocab_size):
        vocab_words.append(word)
    print("词典大小{}".format(len(vocab_words)))

    common_words = []
    print("添加常用单字汉字")
    with open("../data/CommonSingleWords.txt", "r", encoding="utf-8") as fr1:
        lines = fr1.readlines()
        for word in lines:
            common_words.append(word.replace("\n", ""))

    print("添加常用双字汉字")
    with open("../data/CommonDoubleWords.txt", "r", encoding="utf-8") as fr1:
        lines = fr1.readlines()
        for word in lines:
            common_words.append(word.replace("\n", ""))

    vocab_words.extend(common_words)
    print("词典大小{}".format(len(vocab_words)))

    vocab_words = list(set(vocab_words))
    print("去重后词典大小{}".format(len(vocab_words)))

    vocab_words.insert(0, "<pad>")
    vocab_words.insert(1, "<unk>")
    vocab_words.insert(2, "<bos>")
    vocab_words.insert(3, "<eos>")

    print("待保存字典大小{}".format(len(vocab_words)))
    if not os.path.exists("vocab"):
        os.makedirs("vocab")
    with open(ghp.ch_vocab_file, "w", encoding="utf-8") as fw:
        for word in vocab_words:
            fw.write("{}\n".format(word))


def load_ch_vocab():
    vocab = [line.replace("\n", "") for line in open(
        ghp.ch_vocab_file, "r", encoding="utf-8").readlines()]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    print("获取词典大小：{}".format(len(vocab)))
    return word2idx, idx2word


def padder(seq):
    global need_cut_sentences_num
    try:
        seq_pad = np.lib.pad(seq, (0, ghp.max_seq_len - len(seq)),
                             'constant', constant_values=(0, 0))
    except:
        seq_pad = seq[: ghp.max_seq_len - 1]
        seq_pad.append(3)
        need_cut_sentences_num = need_cut_sentences_num + 1
        return seq_pad
    return seq_pad


def create_train_data(ch_sentences):
    # word to idx
    ch2idx, _ = load_ch_vocab()
    idxs = []
    for sentence in tqdm(ch_sentences):
        idx = []
        for word in (sentence + " <eos>").split():
            flag = ch2idx.get(word, 1)
            if flag == 1:
                # idx.append(1)
                for char in word:
                    idx.append(ch2idx.get(char, 1))
            else:
                idx.append(flag)
        idxs.append(idx)
    print("成功！中文句子索引{}条".format(len(idxs)))

    print("(4/4)PAD数据到最大长度{}...".format(ghp.max_seq_len))
    # train_zh_idx = np.zeros([len(ch_list), ghp.max_seq_len], np.int32)
    pool2 = multiprocessing.Pool()

    train_zh_idx = pool2.map(padder, idxs)

    train_zh_idx = np.array(train_zh_idx)

    print("成功！PAD中文句子索引{}条".format(len(idxs)))

    return train_zh_idx


def get_train_data_loader():
    batch_size = ghp.batch_size
    print("#######开始加载训练数据：英文， 中文（已分词）#########")

    print("(1/4)开始创建词典...")
    if os.path.exists(ghp.ch_vocab_file):
        print("字典存在，正在获取...")
        ch2idx, _ = load_ch_vocab()
    else:
        make_ch_vocab(ghp.origin_ch_train_file)
        ch2idx, _ = load_ch_vocab()
    print("成功！词典大小{}".format(ghp.ch_vocab_size))

    print("(2/4)获取训练数据...")
    origin_en_sentences, origin_ch_sentences = load_origin_sentences_data()
    print("成功！中文{}条句子，英文{}条句子".format(
        len(origin_ch_sentences), len(origin_en_sentences)))

    print("(3/4)生成中文句子索引数据...")
    if os.path.exists(ghp.train_ch_idx_file_name + ".npy"):
        train_zh_idx = np.load(ghp.train_ch_idx_file_name + ".npy")
        print("成功！中文句子索引{}条".format(len(train_zh_idx)))
        print("(4/4)PAD数据到最大长度{}...".format(ghp.max_seq_len))
        print("成功！PAD中文句子索引{}条".format(len(train_zh_idx)))
    else:
        train_zh_idx = create_train_data(origin_ch_sentences)
        np.save(ghp.train_ch_idx_file_name, train_zh_idx)

    # check the en and zh has same length
    if len(origin_en_sentences) != len(train_zh_idx):
        raise ValueError(
            "train data is wrong! please make sure the chinese data and english data has the same"
            " length.zh:{} vs en:{}".format(
                len(train_zh_idx),
                len(origin_en_sentences)))

    # make an iterator
    for i in range(int(len(train_zh_idx) / batch_size) + 1):
        batch_en_sentences_data = origin_en_sentences[i * batch_size: min(
            len(train_zh_idx), (i + 1) * batch_size)]
        batch_zh_idx_data = train_zh_idx[i *
                                         batch_size: min(len(train_zh_idx), (i + 1) * batch_size)]
        yield batch_en_sentences_data, batch_zh_idx_data


def get_valid_data_loader():
    batch_size = ghp.batch_size
    print("#######开始加载训练数据：英文， 中文（已分词）#########")

    print("(1/4)开始创建词典...")
    if os.path.exists(ghp.ch_vocab_file):
        ch2idx, _ = load_ch_vocab()
    else:
        make_ch_vocab(ghp.origin_ch_train_file)
        ch2idx, _ = load_ch_vocab()
    print("成功！词典大小{}".format(ghp.ch_vocab_size))

    print("(2/4)获取训练数据...")
    origin_en_sentences, origin_ch_sentences = load_origin_sentences_data()
    print("成功！中文{}条句子，英文{}条句子".format(
        len(origin_ch_sentences), len(origin_en_sentences)))

    print("(3/4)生成中文句子索引数据...")
    if os.path.exists(ghp.train_ch_idx_file_name + ".npy"):
        train_zh_idx = np.load(ghp.train_ch_idx_file_name + ".npy")
        print("成功！中文句子索引{}条".format(len(train_zh_idx)))
        print("(4/4)PAD数据到最大长度{}...".format(ghp.max_seq_len))
        print("成功！PAD中文句子索引{}条".format(len(train_zh_idx)))
    else:
        train_zh_idx = create_train_data(origin_ch_sentences, ch2idx)
        np.save(ghp.train_ch_idx_file_name, train_zh_idx)

    # check the en and zh has same length
    if len(origin_en_sentences) != len(train_zh_idx):
        raise ValueError(
            "train data is wrong! please make sure the chinese data and english data has the same"
            " length.zh:{} vs en:{}".format(
                len(train_zh_idx),
                len(origin_en_sentences)))

    # make an iterator
    for i in range(int(len(train_zh_idx) / batch_size) + 1):
        batch_en_sentences_data = origin_en_sentences[i * batch_size: min(
            len(train_zh_idx), (i + 1) * batch_size)]
        batch_zh_idx_data = train_zh_idx[i *
                                         batch_size: min(len(train_zh_idx), (i + 1) * batch_size)]
        yield batch_en_sentences_data, batch_zh_idx_data


def get_test_data_loader():
    batch_size = ghp.batch_size
    print("#######开始加载训练数据：英文， 中文（已分词）#########")

    print("(1/4)开始创建词典...")
    if os.path.exists(ghp.ch_vocab_file):
        ch2idx, _ = load_ch_vocab()
    else:
        make_ch_vocab(ghp.origin_ch_train_file)
        ch2idx, _ = load_ch_vocab()
    print("成功！词典大小{}".format(ghp.ch_vocab_size))

    print("(2/4)获取训练数据...")
    origin_en_sentences, origin_ch_sentences = load_origin_sentences_data()
    print("成功！中文{}条句子，英文{}条句子".format(
        len(origin_ch_sentences), len(origin_en_sentences)))

    print("(3/4)生成中文句子索引数据...")
    if os.path.exists(ghp.train_ch_idx_file_name + ".npy"):
        train_zh_idx = np.load(ghp.train_ch_idx_file_name + ".npy")
        print("成功！中文句子索引{}条".format(len(train_zh_idx)))
        print("(4/4)PAD数据到最大长度{}...".format(ghp.max_seq_len))
        print("成功！PAD中文句子索引{}条".format(len(train_zh_idx)))
    else:
        train_zh_idx = create_train_data(origin_ch_sentences, ch2idx)
        np.save(ghp.train_ch_idx_file_name, train_zh_idx)

    # check the en and zh has same length
    if len(origin_en_sentences) != len(train_zh_idx):
        raise ValueError(
            "train data is wrong! please make sure the chinese data and english data has the same"
            " length.zh:{} vs en:{}".format(
                len(train_zh_idx),
                len(origin_en_sentences)))

    # make an iterator
    for i in range(int(len(train_zh_idx) / batch_size) + 1):
        batch_en_sentences_data = origin_en_sentences[i * batch_size: min(
            len(train_zh_idx), (i + 1) * batch_size)]
        batch_zh_idx_data = train_zh_idx[i *
                                         batch_size: min(len(train_zh_idx), (i + 1) * batch_size)]
        yield batch_en_sentences_data, batch_zh_idx_data


def main():
    for i in get_train_data_loader():
        pass

if __name__ == '__main__':
    main()

