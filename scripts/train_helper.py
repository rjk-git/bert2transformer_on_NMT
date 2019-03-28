import sys
sys.path.append("..")
import bert_embedding
from models.Transformer import Transformer
from mxnet import nd
from prepo import get_data_loader2, load_zh_vocab, make_zh_vocab2
from hyperParameters import GetHyperParameters as ghp
import numpy as np
import mxnet as mx

def translate():
    zh2idx, idx2zh = load_zh_vocab()

    # build model and load parameters
    model = Transformer(zh2idx.__len__())
    model.load_parameters("parameters/epoch0_batch5000_loss2.8857500553131104_acc0.42683982849121094.params", ctx=ghp.ctx)

    while True:
        # get input english sentence
        bert = bert_embedding.BertEmbedding(ctx=ghp.ctx)
        input_english = input("请输入英文句子：")
        if input_english == "exit":
            break
        bert_result = bert([input_english])

        # pre-process english sentence
        seq_valid_len = len(bert_result[0][0])
        en_idx = [[1] * seq_valid_len + [0] * (ghp.max_seq_len - seq_valid_len)]
        en_idx_nd = nd.array(en_idx, ctx=ghp.ctx)

        en_emb = []
        # embedding
        for word_emb in bert_result[0][1]:
            en_emb.append(word_emb.tolist())

        # padding
        for n in range(ghp.max_seq_len - seq_valid_len):
            en_emb.append([9e-10] * 768)

        en_emb_nd = nd.array([en_emb], ctx=ghp.ctx)

        # prepare init decoder input
        dec_begin_input = [zh2idx["<bos>"]]
        dec_begin_input.extend([zh2idx["<pad>"]] * (ghp.max_seq_len - 1))
        dec_begin_input = nd.array(dec_begin_input, ctx=ghp.ctx)
        dec_input = nd.expand_dims(dec_begin_input, axis=0)

        # begin predict
        zh_seq_len = 1
        predict_token = []

        while True:
            output = model(en_emb_nd, en_idx_nd, dec_input)
            predict = nd.argmax(nd.softmax(output, axis=-1), axis=-1)
            idx = int(predict[0][zh_seq_len-1].asscalar())
            predict_token.append(idx2zh[idx])

            dec_input[0][zh_seq_len] = idx
            zh_seq_len += 1
            if predict_token[-1] == "<eos>":
                break

        print("translate:", "".join(predict_token))


def _init_position_weight():
    position_enc = np.arange(ghp.max_seq_len).reshape((-1, 1)) \
                   / (np.power(10000, (2. / ghp.model_dims) * np.arange(ghp.model_dims).reshape((1, -1))))
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
    return nd.array(position_enc, ctx=ghp.ctx)


if __name__ == '__main__':
    translate()