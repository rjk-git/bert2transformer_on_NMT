import sys
sys.path.append("..")
import numpy as np
from models.Transformer import Transformer
from mxnet import gluon
from mxnet.gluon import loss as gloss
from mxnet import autograd, nd, init
from prepo import get_data_loader, load_ch_vocab, make_ch_vocab
from hyperParameters import GetHyperParameters as ghp
import os
import bert_embedding
from mxboard import *
import mxnet as mx


sw = SummaryWriter(logdir='./logs', flush_secs=5)


def main():
    # build model
    model = Transformer(ghp.ch_vocab_size + 4)
    # model.initialize(init=init.Xavier(), force_reinit=True, ctx=ghp.ctx)
    model.load_parameters("parameters/epoch1_batch15000_loss0.43608570098876953_acc0.8996627330780029.params",
                          ctx=ghp.ctx)

    # train and valid
    train_and_valid(model)


def get_learning_rate(learning_rate, hidden_size, learning_rate_warmup_steps, global_step):
    warmup_steps = float(learning_rate_warmup_steps)
    step = float(global_step)

    learning_rate = (hidden_size ** -0.5) * learning_rate

    learning_rate = np.minimum(1.0, step / warmup_steps) * learning_rate

    learning_rate = (np.maximum(step, warmup_steps) ** -0.5) * learning_rate

    return learning_rate

def train_and_valid(transformer_model):
    loss = gloss.SoftmaxCrossEntropyLoss()
    bert = bert_embedding.BertEmbedding(ctx=ghp.ctx)
    global_step = 0
    for epoch in range(ghp.epoch_num):
        train_data_loader = get_data_loader()
        print("********开始训练********")

        learning_rate = get_learning_rate(ghp.learning_rate, ghp.model_dim, ghp.learning_rate_warmup_steps,
                                          global_step)
        optimizer = mx.optimizer.Adam(learning_rate=learning_rate, beta1=ghp.optimizer_adam_beta1,
                                      beta2=ghp.optimizer_adam_beta2, epsilon=ghp.optimizer_adam_epsilon)

        model_trainer = gluon.Trainer(transformer_model.collect_params(), optimizer)

        count = 0
        last_acc = 0
        for en_sentences, zh_idxs in train_data_loader:
            learning_rate = get_learning_rate(ghp.learning_rate, ghp.model_dim, ghp.learning_rate_warmup_steps,
                                              global_step)
            count += 1
            print("现在是第{}个epoch（总计{}个epoch），第{}批数据。(lr:{}s)"
                  .format(epoch + 1, ghp.epoch_num, count, model_trainer.learning_rate))
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
                loss_mean, acc = batch_loss(transformer_model, en_sentences, x_en_emb, x_en_idx, y_zh_idx, loss)
            loss_scalar = loss_mean.asscalar()
            acc_scalar = acc.asscalar()
            sw.add_scalar(tag='cross_entropy', value=loss_scalar, global_step=global_step)
            sw.add_scalar(tag='acc', value=acc_scalar, global_step=global_step)
            global_step += 1
            loss_mean.backward()
            model_trainer.set_learning_rate(learning_rate)
            if acc_scalar > 0.3:
                acc_diff = acc_scalar - last_acc
                last_acc = acc_scalar
                if acc_diff > 0.1:
                    print("上一步acc:{},此步acc:{},差值为{}过大，放弃更新参数。".format(str(last_acc)[:5], str(acc_scalar)[:5], str(acc_diff)[:5]))
                    continue
            model_trainer.step(1)
            print("loss:{0}, acc:{1}".format(str(loss_scalar)[:5], str(acc_scalar)[:5]))
            print("\n")

            if count % 5000 == 0:
                if not os.path.exists("parameters"):
                    os.makedirs("parameters")
                model_params_file = "parameters/" + "epoch{}_batch{}_loss{}_acc{}.params".format(epoch, count, str(loss_scalar)[:5], str(acc_scalar)[:5])
                transformer_model.save_parameters(model_params_file)


def batch_loss(transformer_model, en_sentences, x_en_emb, x_en_idx, y_zh_idx, loss):
    batch_size = x_en_emb.shape[0]
    ch2idx, idx2ch = load_ch_vocab()

    # make [sentence] + [<eos>] ==> [<bos>] + [sentence]
    dec_input_zh_idx = []
    for i in range(batch_size):
        y_zh_idx_np = y_zh_idx[i]
        eos_idx = np.argwhere(y_zh_idx_np == ch2idx["<eos>"])[0]
        y_zh_idx_np = np.delete(y_zh_idx_np, eos_idx, axis=0)
        y_zh_idx_np = np.insert(y_zh_idx_np, 0,  ch2idx["<bos>"], axis=0)
        dec_input_zh_idx.append(y_zh_idx_np.tolist())

    x_en_emb = x_en_emb.as_in_context(ghp.ctx)
    x_en_idx = x_en_idx.as_in_context(ghp.ctx)
    dec_input_zh_idx = nd.array(dec_input_zh_idx, ghp.ctx)

    output = transformer_model(x_en_emb, x_en_idx, dec_input_zh_idx, True)
    predict = nd.argmax(nd.softmax(output, axis=-1), axis=-1)

    print("source:", en_sentences[0])

    label_token = []
    for n in range(len(y_zh_idx[0])):
        label_token.append(idx2ch[int(y_zh_idx[0][n])])
    print("target:", "".join(label_token))

    predict_token = []
    for n in range(len(predict[0])):
        predict_token.append(idx2ch[int(predict[0][n].asscalar())])
    print("predict:", "".join(predict_token))

    y_zh_idx = nd.array(y_zh_idx, ghp.ctx)
    is_target = nd.not_equal(y_zh_idx, 0)
    current = nd.equal(y_zh_idx, predict) * is_target
    acc = nd.sum(current) / nd.sum(is_target)

    l = loss(output, y_zh_idx)
    l_mean = nd.sum(l) / batch_size

    return l_mean, acc

if __name__ == "__main__":
    main()
