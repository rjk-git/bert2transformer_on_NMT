import mxnet as mx


class GetHyperParameters(object):
    # data
    origin_en_train_file = "./temp_data/origin.en.sentences.train.txt"
    origin_zh_train_file = "./temp_data/origin.zh.sentences.train.txt"

    origin_en_dev_file = "./temp_data/origin.en.sentences.dev.txt"
    origin_zh_dev_file = "./temp_data/origin.zh.sentences.dev.txt"

    origin_en_test_file = "./temp_data/origin.en.sentences.test.txt"
    origin_zh_test_file = "./temp_data/origin.zh.sentences.test.txt"

    handled_en_train_file = "./temp_data/handled.en.sentences.train.txt"
    handled_zh_train_file = "./temp_data/handled.zh.sentences.train.txt"

    handled_en_dev_file = "./temp_data/handled.en.sentences.dev.txt"
    handled_zh_dev_file = "./temp_data/handled.zh.sentences.dev.txt"

    handled_en_test_file = "./temp_data/handled.en.sentences.test.txt"
    handled_zh_test_file = "./temp_data/handled.zh.sentences.test.txt"

    handled_zh_idx_file_name = "./temp_data/handled.zh.idx.train"

    zh_vocab_file = "./vocab/zh_vocab.tsv"
    en_vocab_file = "./vocab/en_vocab.tsv"

    # process corpus params
    min_count_vocab_size = 20
    max_seq_len = 25
    en_vocab_size = 200000
    zh_vocab_size = 250000

    # model hyper params
    layer_num = 6
    head_num = 8
    model_dim = 512
    c_dim = 64
    ffn_dim = 2048
    dropout = 0.1
    epsilon = 1e-8

    # train params
    lr = 0.0001
    batch_size = 128
    epoch_num = 25
    ctx = mx.gpu()