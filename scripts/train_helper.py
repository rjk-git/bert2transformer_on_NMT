import sys
sys.path.append("..")
import bert_embedding

from mxnet import nd
from models.Transformer import Transformer

from prepo import get_train_data_loader, load_ch_vocab, make_ch_vocab
from hyperParameters import GetHyperParameters as ghp


def translate():
    zh2idx, idx2zh = load_ch_vocab()

    # build model and load parameters
    model = Transformer(zh2idx.__len__())
    model.load_parameters("parameters/re_epoch0_batch80000_loss1.621_acc0.355.params", ctx=ghp.ctx)

    while True:
        # get input english sentence
        bert = bert_embedding.BertEmbedding(ctx=ghp.ctx)
        input_english = input("input english sentence：")
        if input_english == "exit":
            break

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
        predict = nd.array([[2] + [0] * (ghp.max_seq_len-1)], ctx=ghp.ctx)

        for n in range(1, ghp.max_seq_len):
            output = model(en_emb, en_idx, predict, False)
            predict_ = nd.argmax(nd.softmax(output, axis=-1), axis=-1)
            predict[0][n] = predict_[0][n-1]

        predict_token = [idx2zh[int(idx.asscalar())] for idx in predict[0]]
        print("translate:", "".join(predict_token))


def compute_bleu(reference, candidate):
    smooth = SmoothingFunction()
    score = corpus_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth.method7)
    print(score)

if __name__ == '__main__':
    translate()