import sys
sys.path.append("..")

import os
import mxnet as mx
import numpy as np
from mxboard import *
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss
from hyperParameters import GetHyperParameters as ghp
from models.Transformer import Transformer
from prepo import get_train_data_loader, load_ch_vocab


sw = SummaryWriter(logdir='./logs', flush_secs=5)


def main():
    # build model
    word2idx, _ = load_ch_vocab()
    model = Transformer(len(word2idx))

    # model.load_parameters("./parameters/*****.params", ctx=ghp.ctx)

    model.decoder.initialize(init=init.Xavier(), ctx=ghp.ctx)
    model.en_input_dense.initialize(init=init.Xavier(), ctx=ghp.ctx)
    model.linear.initialize(init=init.Xavier(), ctx=ghp.ctx)

    # train and valid
    train_and_valid(model)


def get_learning_rate(step_num, warm_up_step=4000, d_model=ghp.model_dim):
    learning_rate = pow(d_model, -0.5) * min(pow(step_num, -0.5),
                                             (step_num * pow(warm_up_step, -1.5)))
    return learning_rate


def train_and_valid(transformer_model):
    loss = gloss.SoftmaxCrossEntropyLoss()
    global_step = 1
    learning_rate = get_learning_rate(global_step)
    optimizer = mx.optimizer.Adam(learning_rate=learning_rate)

    bert_optimizer = mx.optimizer.Adam(learning_rate=2e-5)

    bert_trainer = gluon.Trainer(
        transformer_model.encoder.collect_params(), bert_optimizer)
    model_trainer = gluon.Trainer(transformer_model.collect_params(
        select="decoder0_*|en_input_dense0_*|linear0_*"), optimizer)

    for epoch in range(ghp.epoch_num):
        train_data_loader = get_train_data_loader()
        print("********开始训练********")
        count = 0
        for en_sentences, zh_idxs in train_data_loader:
            count += 1
            print("现在是第{}个epoch（总计{}个epoch），第{}批数据。(lr:{}s)"
                  .format(epoch + 1, ghp.epoch_num, count, model_trainer.learning_rate))

            y_zh_idx = zh_idxs

            with autograd.record():
                loss_mean, acc = batch_loss(
                    transformer_model, en_sentences, y_zh_idx, loss)
            loss_scalar = loss_mean.asscalar()
            acc_scalar = acc.asscalar()
            sw.add_scalar(tag='cross_entropy', value=loss_scalar,
                          global_step=global_step)
            sw.add_scalar(tag='acc', value=acc_scalar, global_step=global_step)
            global_step += 1
            loss_mean.backward()
            learning_rate = get_learning_rate(global_step)
            model_trainer.set_learning_rate(learning_rate)
            model_trainer.step(1)
            bert_trainer.step(1)

            print("loss:{0}, acc:{1}".format(
                str(loss_scalar)[:5], str(acc_scalar)[:5]))
            print("\n")

            if count % 5000 == 0:
                if not os.path.exists("parameters"):
                    os.makedirs("parameters")
                model_params_file = "parameters/" + "re3_epoch{}_batch{}_loss{}_acc{}.params".format(
                    epoch, count, str(loss_scalar)[:5], str(acc_scalar)[:5])
                transformer_model.save_parameters(model_params_file)


def batch_loss(transformer_model, en_sentences, y_zh_idx, loss):
    batch_size = y_zh_idx.shape[0]
    ch2idx, idx2ch = load_ch_vocab()

    y_zh_idx_nd = nd.array(y_zh_idx, ctx=ghp.ctx)
    dec_input_zh_idx = nd.concat(nd.ones(
        shape=y_zh_idx_nd[:, :1].shape, ctx=ghp.ctx) * 2, y_zh_idx_nd[:, :-1], dim=1)

    output = transformer_model(en_sentences, dec_input_zh_idx, True)
    predict = nd.argmax(nd.softmax(output, axis=-1), axis=-1)
    is_target = nd.not_equal(y_zh_idx_nd, 0)

    # print("input_idx:", dec_input_right_idx[0])
    # print("predict_idx:", predict[0])
    print("source :", en_sentences[0])

    label_token = []
    for n in range(int(nd.sum(is_target, axis=-1)[0].asscalar())):
        label_token.append(idx2ch[int(y_zh_idx[0][n])])
    print("target :", " ".join(label_token))

    predict_token = []
    for n in range(ghp.max_seq_len):
        predict_token.append(idx2ch[int(predict[0][n].asscalar())])
    print("predict:", " ".join(predict_token))

    current = nd.equal(y_zh_idx_nd, predict) * is_target
    acc = nd.sum(current) / nd.sum(is_target)

    # l_mean = my_loss(output, idx_right_nd)
    l = loss(output, y_zh_idx_nd)
    l_mean = nd.sum(l) / batch_size

    return l_mean, acc


if __name__ == "__main__":
    main()
