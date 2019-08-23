import argparse
import math
import os
import sys
sys.path.append("..")

import gluonnlp
import mxnet as mx
import numpy as np
from gluonnlp.data import train_valid_split
from gluonnlp.model import BeamSearchScorer
from mxboard import *
from mxnet import autograd, gluon, init, nd
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from numpy import random
from pypinyin import pinyin
from tqdm import tqdm

from data.scripts.dataloader import DatasetAssiantTransformer, MTDataLoader
from data.scripts.dataset import MTDataset
from models.MaskedCELoss import MaskedCELoss
from models.MTModel_Hybird import Transformer as MTModel_Hybird
from utils import config_logger

np.random.seed(100)
random.seed(100)
mx.random.seed(10000)
BOS = "[unused16]"
EOS = "[unused17]"


def train_and_valid(en_bert, mt_model, en_vocab, ch_vocab, train_dataiter, dev_dataiter, trainer, en_finetune_trainer, epochs, loss_func, ctx, lr, batch_size, params_save_step, params_save_path_root, eval_step, log_step, check_step, label_smooth, logger, num_train_examples, warmup_ratio):
    batches = len(train_dataiter)

    num_train_steps = int(num_train_examples / batch_size * epochs)
    num_warmup_steps = int(num_train_steps * warmup_ratio)
    global_step = 0
    dev_bleu_score = 0

    for epoch in range(epochs):
        for trans, aim, label, trans_valid_len, aim_valid_len in train_dataiter:
            if global_step < num_warmup_steps:
                new_lr = lr * global_step / num_warmup_steps
            else:
                non_warmup_steps = global_step - num_warmup_steps
                offset = non_warmup_steps / (num_train_steps - num_warmup_steps)
                new_lr = lr - offset * lr
            trainer.set_learning_rate(new_lr)

            trans = trans.as_in_context(ctx)
            aim = aim.as_in_context(ctx)
            label = label.as_in_context(ctx)
            trans_valid_len = trans_valid_len.as_in_context(ctx)
            trans_token_type = nd.zeros_like(trans, ctx=ctx)

            aim_mask = nd.not_equal(aim, ch_vocab(ch_vocab.padding_token))

            if label_smooth:
                eps = 0.1
                num_class = len(ch_vocab.idx_to_token)
                one_hot = nd.one_hot(label, num_class)
                one_hot_label = one_hot * (1 - eps) + (1 - one_hot) * eps / num_class

            with autograd.record():
                en_bert_outputs = en_bert(trans, trans_token_type, trans_valid_len)
                mt_outputs = mt_model(en_bert_outputs, trans, aim)
                loss_mean = loss_func(mt_outputs, one_hot_label, aim_mask)

            loss_mean.backward()
            loss_scalar = loss_mean.asscalar()

            trainer.step(1)
            en_finetune_trainer.step(1)

            if global_step and global_step % log_step == 0:
                predicts = nd.argmax(nd.softmax(mt_outputs, axis=-1), axis=-1)
                correct = nd.equal(label, predicts)
                accuracy = (nd.sum(correct * aim_mask) / nd.sum(aim_mask)).asscalar()
                logger.info("epoch:{}, batch:{}/{}, bleu:{}, acc:{}, loss:{}, (lr:{}s)"
                            .format(epoch, global_step % batches, batches, dev_bleu_score, accuracy, loss_scalar, trainer.learning_rate))

            if global_step and global_step % check_step == 0:
                predicts = nd.argmax(nd.softmax(mt_outputs, axis=-1), axis=-1)
                refer_sample = trans.asnumpy().tolist()
                label_sample = label.asnumpy().tolist()
                pred_sample = predicts.asnumpy().tolist()
                logger.info("train sample:")
                logger.info("refer  :{}".format(
                    " ".join([en_vocab.idx_to_token[int(idx)] for idx in refer_sample[0]])).replace(en_vocab.padding_token, ""))
                logger.info("target :{}".format(
                    " ".join([ch_vocab.idx_to_token[int(idx)] for idx in label_sample[0]])).replace(EOS, "[EOS]").replace(ch_vocab.padding_token, ""))
                logger.info("predict:{}".format(
                    " ".join([ch_vocab.idx_to_token[int(idx)] for idx in pred_sample[0]])).replace(EOS, "[EOS]"))

            if global_step and global_step % eval_step == 0:
                dev_bleu_score = eval(en_bert, mt_model, en_vocab,
                                      ch_vocab, dev_dataiter, logger, ctx=ctx)

            if global_step and global_step % params_save_step == 0:
                if not os.path.exists(params_save_path_root):
                    os.makedirs(params_save_path_root)
                model_params_file = params_save_path_root + \
                    "en_bert.ft_step_{}.params".format(global_step)
                en_bert.save_parameters(model_params_file)
                logger.info("{} Save Completed.".format(model_params_file))

                model_params_file = params_save_path_root + \
                    "mt_step_{}.params".format(global_step)
                mt_model.save_parameters(model_params_file)
                logger.info("{} Save Completed.".format(model_params_file))

            global_step += 1


def eval(en_bert, mt_model, en_vocab, ch_vocab, dev_dataiter, logger, ctx):
    references = []
    hypothesis = []
    score = 0
    chencherry = SmoothingFunction()
    for trans, _, label, trans_valid_len, label_valid_len in tqdm(dev_dataiter):
        trans = trans.as_in_context(ctx)
        trans_valid_len = trans_valid_len.as_in_context(ctx)
        batch_size = trans.shape[0]

        trans_token_type = nd.zeros_like(trans)
        en_bert_outputs = en_bert(trans, trans_token_type, trans_valid_len)

        ch_sentences = [BOS]
        aim = ch_vocab[ch_sentences]
        aim = nd.array([aim], ctx=ctx)
        aim = nd.broadcast_axes(aim, axis=0, size=batch_size)

        for n in range(0, args.max_ch_len):
            mt_outputs = mt_model(en_bert_outputs, trans, aim)
            predicts = nd.argmax(nd.softmax(mt_outputs, axis=-1), axis=-1)
            final_predict = predicts[:, -1:]
            aim = nd.concat(aim, final_predict, dim=1)

        label = label.asnumpy().tolist()
        predict_valid_len = nd.sum(nd.not_equal(predicts, ch_vocab(
            ch_vocab.padding_token)), axis=-1).asnumpy().tolist()
        predicts = aim[:, 1:].asnumpy().tolist()
        label_valid_len = label_valid_len.asnumpy().tolist()

        for refer, hypoth, l_v_len, p_v_len in zip(label, predicts, label_valid_len, predict_valid_len):
            l_v_len = int(l_v_len)
            p_v_len = int(p_v_len)
            refer = refer[:l_v_len]
            refer_str = [ch_vocab.idx_to_token[int(idx)] for idx in refer]
            hypoth_str = [ch_vocab.idx_to_token[int(idx)] for idx in hypoth]
            hypoth_str_valid = []
            for token in hypoth_str:
                if token == EOS:
                    hypoth_str_valid.append(token)
                    break
                hypoth_str_valid.append(token)
            references.append(refer_str)
            hypothesis.append(hypoth_str_valid)

    for refer, hypoth in zip(references, hypothesis):
        score += sentence_bleu([refer], hypoth, smoothing_function=chencherry.method1)
    logger.info("dev sample:")
    logger.info("refer :{}".format(" ".join(references[0]).replace(
        EOS, "[EOS]").replace(ch_vocab.padding_token, "")))
    logger.info("hypoth:{}".format(" ".join(hypothesis[0]).replace(
        EOS, "[EOS]")))
    return score / len(references)


def main(args):
    # init some setting
    # config logging
    log_path = os.path.join(args.log_root, '{}.log'.format(args.model_name))
    logger = config_logger(log_path)

    gpu_idx = args.gpu
    if not gpu_idx:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(gpu_idx - 1)
    logger.info("Using ctx: {}".format(ctx))

    # Loading vocab and model
    en_bert, en_vocab = gluonnlp.model.get_model(args.bert_model,
                                                 dataset_name=args.en_bert_dataset,
                                                 pretrained=True,
                                                 ctx=ctx,
                                                 use_pooler=False,
                                                 use_decoder=False,
                                                 use_classifier=False)
    _, ch_vocab = gluonnlp.model.get_model(args.bert_model,
                                           dataset_name=args.ch_bert_dataset,
                                           pretrained=True,
                                           ctx=ctx,
                                           use_pooler=False,
                                           use_decoder=False,
                                           use_classifier=False)

    mt_model = MTModel_Hybird(en_vocab=en_vocab, ch_vocab=ch_vocab, embedding_dim=args.mt_emb_dim, model_dim=args.mt_model_dim, head_num=args.mt_head_num,
                              layer_num=args.mt_layer_num, ffn_dim=args.mt_ffn_dim, dropout=args.mt_dropout, att_dropout=args.mt_att_dropout, ffn_dropout=args.mt_ffn_dropout, ctx=ctx)
    logger.info("Model Creating Completed.")

    # init or load params for model
    mt_model.initialize(init.Xavier(), ctx)

    # en_bert.load_parameters(args.en_bert_model_params_path, ctx=ctx)
    # mt_model.load_parameters(args.mt_model_params_path, ctx=ctx)
    logger.info("Parameters Initing and Loading Completed")

    en_bert.hybridize()
    mt_model.hybridize()

    # Loading dataloader
    assiant = DatasetAssiantTransformer(
        en_vocab=en_vocab, ch_vocab=ch_vocab, max_en_len=args.max_en_len, max_ch_len=args.max_ch_len)
    dataset = MTDataset(args.train_en_data_path, args.train_ch_data_path)
    train_dataset, dev_dataset = train_valid_split(dataset, valid_ratio=args.valid_ratio)
    train_dataiter = MTDataLoader(train_dataset, batch_size=args.batch_size,
                                  assiant=assiant, shuffle=True).dataiter
    dev_dataiter = MTDataLoader(dev_dataset, batch_size=int(args.batch_size / 2),
                                assiant=assiant, shuffle=True).dataiter
    logger.info("Data Loading Completed")

    # build trainer
    en_finetune_trainer = gluon.Trainer(en_bert.collect_params(),
                                        args.optimizer, {"learning_rate": args.finetune_lr})
    trainer = gluon.Trainer(mt_model.collect_params(), args.optimizer,
                            {"learning_rate": args.train_lr})

    # loss function
    if args.label_smooth:
        loss_func = MaskedCELoss(sparse_label=False)
    else:
        loss_func = MaskedCELoss()

    logger.info("## Trainning Start ##")
    train_and_valid(
        en_bert=en_bert, mt_model=mt_model, en_vocab=en_vocab, ch_vocab=ch_vocab,
        train_dataiter=train_dataiter, dev_dataiter=dev_dataiter, trainer=trainer, en_finetune_trainer=en_finetune_trainer, epochs=args.epochs,
        loss_func=loss_func, ctx=ctx, lr=args.train_lr, batch_size=args.batch_size, params_save_step=args.params_save_step,
        params_save_path_root=args.params_save_path_root, eval_step=args.eval_step, log_step=args.log_step, check_step=args.check_step,
        label_smooth=args.label_smooth, logger=logger, num_train_examples=len(train_dataset), warmup_ratio=args.warmup_ratio
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Use Bert's Outputs as Transformer's Encoder Outputs, the Decoder is a Norm Transformer's Decoder.")
    parser.add_argument("--model_name", type=str, default="BertBasedEn2ZhModel")
    parser.add_argument("--train_en_data_path", type=str, default="")
    parser.add_argument("--train_ch_data_path", type=str, default="")
    parser.add_argument("--bert_model", type=str,
                        default="bert_12_768_12")
    parser.add_argument("--en_bert_dataset", type=str,
                        default="openwebtext_book_corpus_wiki_en_uncased")
    parser.add_argument("--ch_bert_dataset", type=str,
                        default="wiki_cn_cased")
    parser.add_argument("--en_bert_model_params_path", type=str,
                        default="")
    parser.add_argument("--mt_model_params_path", type=str,
                        default="")
    parser.add_argument("--gpu", type=int,
                        default=1, help='which gpu to use for finetuning. CPU is used if set 0.')
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--train_lr", type=float, default=0.0002)
    parser.add_argument("--finetune_lr", type=float, default=2e-5)
    parser.add_argument("--label_smooth", type=bool, default=True)
    parser.add_argument("--batch_size", type=int,
                        default=32)
    parser.add_argument("--epochs", type=int,
                        default=1)
    parser.add_argument("--valid_ratio", type=int,
                        default=0.00005)
    parser.add_argument("--log_root", type=str, default="../logs/")
    parser.add_argument("--log_step", type=int, default=10)
    parser.add_argument("--eval_step", type=int, default=1000)
    parser.add_argument("--check_step", type=int, default=100)
    parser.add_argument("--params_save_step", type=int, default=5000)
    parser.add_argument("--params_save_path_root", type=str, default="../parameters/")
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='ratio of warmup steps that linearly increase learning rate from '
                        '0 to target learning rate. default is 0.1')
    parser.add_argument("--max_en_len", type=int,
                        default=128)
    parser.add_argument("--max_ch_len", type=int,
                        default=128)
    # translation model parameters setting
    parser.add_argument("--mt_model_dim", type=int,
                        default=768)
    parser.add_argument("--mt_emb_dim", type=int,
                        default=768)
    parser.add_argument("--mt_head_num", type=int,
                        default=8)
    parser.add_argument("--mt_layer_num", type=int,
                        default=6)
    parser.add_argument("--mt_ffn_dim", type=int,
                        default=2048)
    parser.add_argument("--mt_dropout", type=float,
                        default=0.1)
    parser.add_argument("--mt_ffn_dropout", type=float,
                        default=0.1)
    parser.add_argument("--mt_att_dropout", type=float,
                        default=0.1)

    args = parser.parse_args()

    main(args)
