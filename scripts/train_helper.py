import sys
sys.path.append("..")
import bert_embedding
from models.Transformer import Transformer
from mxnet import nd
from prepo import get_data_loader, load_ch_vocab, make_ch_vocab
from hyperParameters import GetHyperParameters as ghp
import numpy as np
import mxnet as mx


def translate():
    zh2idx, idx2zh = load_ch_vocab()

    # build model and load parameters
    model = Transformer(zh2idx.__len__())
    model.load_parameters("parameters/epoch0_batch30000_loss1.1893165111541748_acc0.6940639019012451.params", ctx=ghp.ctx)

    while True:
        # get input english sentence
        bert = bert_embedding.BertEmbedding(ctx=ghp.ctx)
        input_english = input("请输入英文句子：")
        if input_english == "exit":
            break

        # 用bert对输入的英文句子进行初始化
        result = bert([input_english])
        all_sentences_emb = []
        all_sentences_idx = []
        real_batch_size = len([input_english])
        for i in range(real_batch_size):
            one_sent_emb = []

            seq_valid_len = len(result[i][0])
            one_sent_idx = [1] * seq_valid_len + [0] * (ghp.max_seq_len - seq_valid_len)

            # embedding
            for word_emb in result[i][1]:
                one_sent_emb.append(word_emb.tolist())

            # padding
            for n in range(ghp.max_seq_len - seq_valid_len):
                one_sent_emb.append([9e-10] * 768)

            all_sentences_emb.append(one_sent_emb)
            all_sentences_idx.append(one_sent_idx)

        # x_en_emb shape: (1, max_seq_len, 768)
        # x_en_idx shape: (1, max_seq_len)
        en_emb = nd.array(all_sentences_emb, ctx=ghp.ctx)
        en_idx = nd.array(all_sentences_idx, ctx=ghp.ctx)

        # x_en_idx shape: (1, max_seq_len)
        predict = nd.array([[2] + [0] * (ghp.max_seq_len - 1)], ctx=ghp.ctx)

        # test:
        # test_sentence = "表演 的 压轴戏 是 闹剧 版 《 天鹅湖 》 ， 男女 小 人们 身着 粉红色 的 芭蕾舞 裙 扮演 小天鹅 。"
        # test_idx = [zh2idx[word] for word in test_sentence.split()]
        # test_idx_padded = test_idx + [0] * (ghp.max_seq_len - len(test_idx))
        # predict = nd.array([test_idx_padded], ctx=ghp.ctx)

        for n in range(ghp.max_seq_len):
            output = model(en_emb, en_idx, predict, False)
            predict_ = nd.argmax(nd.softmax(output, axis=-1), axis=-1)
            predict[0][n] = predict_[0][n]
        predict_token = [idx2zh[int(idx.asscalar())] for idx in predict[0]]
        print("translate:", "".join(predict_token))


def _init_position_weight():
    position_enc = np.arange(ghp.max_seq_len).reshape((-1, 1)) \
                   / (np.power(10000, (2. / ghp.model_dims) * np.arange(ghp.model_dims).reshape((1, -1))))
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
    return nd.array(position_enc, ctx=ghp.ctx)


if __name__ == '__main__':
    translate()