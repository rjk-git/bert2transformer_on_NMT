import mxnet as mx


class GetHyperParameters(object):
    # data
    source_train = "data"

    # process corpus params
    min_count_vocab_size = 1
    max_seq_len = 30
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