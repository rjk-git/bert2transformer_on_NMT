import mxnet as mx


class GetHyperParameters(object):
    # data
    origin_en_train_file = "./data/train.en.sentences"
    origin_ch_train_file = "./data/train.ch.sentences"

    origin_en_dev_file = "./data/dev.en.sentences"
    origin_ch_dev_file = "./data/dev.ch.sentences"

    origin_en_test_file = "./data/test.en.sentences"
    origin_ch_test_file = "./data/test.ch.sentences"

    train_ch_idx_file_name = "./data/train.ch.idx"

    ch_vocab_file = "./vocab/ch_vocab.tsv"
    en_vocab_file = "./vocab/en_vocab.tsv"

    # process corpus params
    min_count_vocab_size = 20
    max_seq_len = 25
    en_vocab_size = 100000
    ch_vocab_size = 120000

    # model hyper params
    layer_num = 6
    head_num = 8
    model_dim = 512
    c_dim = 64
    ffn_dim = 2048
    dropout = 0.3
    ffn_dropout = 0.3
    attention_epsilon = 1e-9
    norm_epsilon = 1e-6

    # train params
    batch_size = 64
    epoch_num = 25
    ctx = mx.gpu()

    learning_rate = 16.0
    learning_rate_decay_rate = 1.0
    learning_rate_warmup_steps = 216000

    optimizer_adam_beta1 = 0.9
    optimizer_adam_beta2 = 0.997
    optimizer_adam_epsilon = 1e-09