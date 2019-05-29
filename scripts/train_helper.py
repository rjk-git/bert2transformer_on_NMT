import sys
sys.path.append("..")
import bert_embedding

from mxnet import nd
from models.Transformer import Transformer

from prepo import get_train_data_loader, load_ch_vocab, make_ch_vocab
from hyperParameters import GetHyperParameters as ghp


def translate_by_debug():
    zh2idx, idx2zh = load_ch_vocab()

    # build model and load parameters
    model = Transformer(zh2idx.__len__())
    model.load_parameters("parameters/5_24_epoch0_batch90000_loss0.932_acc0.761.params", ctx=ghp.ctx)

    while True:
        input_english = input("input english sentence：")
        if input_english == "exit":
            break

        # x_en_idx shape: (1, max_seq_len)
        predict = nd.array([[2] + [0] * (ghp.max_seq_len-1)], ctx=ghp.ctx)
        output = model([input_english], predict, True)
        predict_ = nd.argmax(nd.softmax(output, axis=-1), axis=-1)
        predict_token = [idx2zh[int(idx.asscalar())] for idx in predict_[0]]
        print("结果     ：{}".format(" ".join(predict_token)))

        predict = nd.array([[2] + [4536] + [123] + [56543] + [0] * (ghp.max_seq_len-4)], ctx=ghp.ctx)
        output = model([input_english], predict, True)
        predict_ = nd.argmax(nd.softmax(output, axis=-1), axis=-1)
        predict_token = [idx2zh[int(idx.asscalar())] for idx in predict_[0]]
        print("结果     ：{}".format(" ".join(predict_token)))


def translate(model, input_english):
    ch2idx, idx2ch = load_ch_vocab()

    # x_en_idx shape: (1, max_seq_len)
    predict = nd.array([[2] + [0] * (ghp.max_seq_len - 1)], ctx=ghp.ctx)

    for n in range(1, ghp.max_seq_len):
        output = model([input_english], predict, True)
        predict_ = nd.argmax(nd.softmax(output, axis=-1), axis=-1)
        predict[0][n] = predict_[0][n - 1]
        if predict_[0][n - 1] == 3:
            break

    predict_token = [idx2ch[int(idx.asscalar())] for idx in predict[0]]
    return "".join(predict_token)


if __name__ == '__main__':
    translate_by_debug()
