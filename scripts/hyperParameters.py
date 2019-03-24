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
    max_seq_len = 20
    en_vocab_size = 80000
    zh_vocab_size = 100000

    # model hyper params
    layer_nums = 6
    head_nums = 8
    model_dims = 512
    k_dims = 64
    ffn_dims = 2048
    dropout = 0.1

    # train params
    lr = 0.00001
    batch_size = 64
    epoch_nums = 25
    ctx = mx.gpu()