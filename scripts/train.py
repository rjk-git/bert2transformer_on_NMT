import sys
sys.path.append("..")
import numpy as np
from models.Transformer import Transformer
from mxnet import gluon
from mxnet.gluon import loss as gloss
from mxnet import autograd
from mxnet import nd
from prepo import get_data_loader, load_zh_vocab
from hyperParameters import GetHyperParameters as ghp
import os
import bert_embedding
from mxboard import *

sw = SummaryWriter(logdir='./logs', flush_secs=5)

def main():
    # get dataSet
    train_data_loader = get_data_loader()
    zh2idx, _ = load_zh_vocab()

    # build model
    model = Transformer(zh2idx.__len__())
    model.initialize(ctx=ghp.ctx)
    position_weight = _init_position_weight()

    # Attach a positional vector to the position embedding layer
    model.decoder.position_embedding.position_embedding.weight.set_data(position_weight)
    model.decoder.position_embedding.collect_params().setattr('grad_req', 'null')

    # train and valid
    train_and_valid(model, train_data_loader)


def train_and_valid(transformer_model, data_loader_train):
    lr_cut = 0.9
    loss = gloss.SoftmaxCrossEntropyLoss()
    bert = bert_embedding.BertEmbedding()
    global_step = 0
    for epoch in range(ghp.epoch_nums):
        # lr = ghp.lr * lr_cut
        model_trainer = gluon.Trainer(transformer_model.collect_params(), 'adam', {"learning_rate" : ghp.lr})
        count = 0

        for en_sentences, zh_idxs in data_loader_train:
            count += 1
            print("现在是第{}个epoch（总计{}个epoch），第{}批数据。(lr:{}s)"
                  .format(epoch + 1, ghp.epoch_nums, count, model_trainer.learning_rate))

            result = bert(en_sentences)
            all_sentences_emb = []
            all_sentences_idx = []
            real_batch_size = len(en_sentences)
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

            x_en_emb = nd.array(all_sentences_emb)
            x_en_idx = nd.array(all_sentences_idx)

            y_zh_idx = zh_idxs

            with autograd.record():
                loss_mean, acc = batch_loss(transformer_model, x_en_emb, x_en_idx, y_zh_idx, loss)
            sw.add_scalar(tag='cross_entropy', value=loss_mean.asscalar(), global_step=global_step)
            global_step += 1
            loss_mean.backward()
            model_trainer.step(x_en_emb.shape[0])
            print("loss:{0}, acc:{1}".format(loss_mean.asscalar(), acc.asscalar()))
            print("\n")

        if not os.path.exists("parameter"):
            os.makedirs("parameter")
        model_params_file = "parameter/epoch" + str(epoch) + ".params"
        transformer_model.save_parameters(model_params_file)


def batch_loss(transformer_model, x_en_emb, x_en_idx, y_zh_idx, loss):
    batch_size = x_en_emb.shape[0]
    zh2idx, idx2zh = load_zh_vocab()

    # make [sentence] + [<eos>] ==> [<bos>] + [sentence]
    dec_input_zh_idx = []
    for i in range(batch_size):
        y_zh_idx_np = y_zh_idx[i]
        eos_idx = np.argwhere(y_zh_idx_np == zh2idx["<eos>"])[0]
        y_zh_idx_np = np.delete(y_zh_idx_np, eos_idx, axis=0)
        y_zh_idx_np = np.insert(y_zh_idx_np, 0,  zh2idx["<bos>"], axis=0)
        dec_input_zh_idx.append(y_zh_idx_np.tolist())

    x_en_emb = x_en_emb.as_in_context(ghp.ctx)
    x_en_idx = x_en_idx.as_in_context(ghp.ctx)
    dec_input_zh_idx = nd.array(dec_input_zh_idx, ghp.ctx)

    output, dec_self_attn, context_attn = transformer_model(x_en_emb, x_en_idx, dec_input_zh_idx)
    predict = nd.argmax(nd.softmax(output, axis=-1), axis=-1)

    label_token = []
    for n in range(len(y_zh_idx[0])):
        label_token.append(idx2zh[int(y_zh_idx[0][n])])
    print("src:", "".join(label_token))

    predict_token = []
    for n in range(len(predict[0])):
        predict_token.append(idx2zh[int(predict[0][n].asscalar())])
    print("predict", "".join(predict_token))

    # translate(transformer_model, "Don 't run away with and idea that you didn 't think through.")

    y_zh_idx = nd.array(y_zh_idx, ghp.ctx)
    is_target = nd.not_equal(y_zh_idx, 0)
    current = nd.equal(y_zh_idx, predict) * is_target
    acc = nd.sum(current) / nd.sum(is_target)

    l = loss(output, y_zh_idx)
    l_mean = nd.sum(l) / batch_size

    return l_mean, acc


def _init_position_weight():
    position_enc = np.arange(ghp.max_seq_len).reshape((-1, 1)) \
                   / (np.power(10000, (2. / ghp.model_dims) * np.arange(ghp.model_dims).reshape((1, -1))))
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
    return nd.array(position_enc, ctx=ghp.ctx)


if __name__ == "__main__":
    main()
