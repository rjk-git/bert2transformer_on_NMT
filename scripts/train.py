import argparse
import math
import os
import sys
sys.path.append("..")

import gluonnlp
import mxnet as mx
import numpy as np
from data.scripts.dataloader import DatasetAssiantTransformer, MTDataLoader
from data.scripts.dataset import MTDataset
from gluonnlp.data import train_valid_split
from gluonnlp.model import BeamSearchScorer
from models.MaskedCELoss import MaskedCELoss
from models.MTModel_Hybird import Transformer as MTModel_Hybird
from mxnet import autograd, gluon, init, nd
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from numpy import random
from tensorboardX import SummaryWriter
from tqdm import tqdm

from utils import config_logger



np.random.seed(100)
random.seed(100)
mx.random.seed(10000)
BOS = "[CLS]"
EOS = "[SEP]"
writer = SummaryWriter("../runs")


def train_and_valid(src_bert,
                    mt_model,
                    src_vocab,
                    tgt_vocab,
                    train_dataiter,
                    dev_dataiter,
                    trainer,
                    finetune_trainer,
                    epochs,
                    loss_func,
                    ctx,
                    lr,
                    batch_size,
                    params_save_path_root,
                    eval_step,
                    log_step,
                    check_step,
                    label_smooth,
                    logger,
                    num_train_examples,
                    warmup_ratio):
    batches = len(train_dataiter)

    num_train_steps = int(num_train_examples / batch_size * epochs)
    num_warmup_steps = int(num_train_steps * warmup_ratio)
    global_step = 0
    dev_bleu_score = 0

    for epoch in range(epochs):
        for src, tgt, label, src_valid_len, tgt_valid_len in train_dataiter:
            # learning rate strategy
            if global_step < num_warmup_steps:
                new_lr = lr * global_step / num_warmup_steps
            else:
                non_warmup_steps = global_step - num_warmup_steps
                offset = non_warmup_steps / \
                    (num_train_steps - num_warmup_steps)
                new_lr = lr - offset * lr
            trainer.set_learning_rate(new_lr)

            src = src.as_in_context(ctx)
            tgt = tgt.as_in_context(ctx)
            label = label.as_in_context(ctx)
            src_valid_len = src_valid_len.as_in_context(ctx)
            src_token_type = nd.zeros_like(src, ctx=ctx)

            tgt_mask = nd.not_equal(tgt, tgt_vocab(tgt_vocab.padding_token))

            if label_smooth:
                eps = 0.1
                num_class = len(tgt_vocab.idx_to_token)
                one_hot = nd.one_hot(label, num_class)
                one_hot_label = one_hot * \
                    (1 - eps) + (1 - one_hot) * eps / num_class

            with autograd.record():
                src_bert_outputs = src_bert(
                    src, src_token_type, src_valid_len)
                mt_outputs = mt_model(src_bert_outputs, src, tgt)
                loss_mean = loss_func(mt_outputs, one_hot_label, tgt_mask)

            loss_mean.backward()
            loss_scalar = loss_mean.asscalar()

            trainer.step(1)
            finetune_trainer.step(1)

            if global_step and global_step % log_step == 0:
                predicts = nd.argmax(nd.softmax(mt_outputs, axis=-1), axis=-1)
                correct = nd.equal(label, predicts)
                accuracy = (nd.sum(correct * tgt_mask) /
                            nd.sum(tgt_mask)).asscalar()
                logger.info("epoch:{}, batch:{}/{}, bleu:{}, acc:{}, loss:{}, (lr:{}s)"
                            .format(epoch, global_step % batches, batches, dev_bleu_score, accuracy, loss_scalar, trainer.learning_rate))

            if global_step and global_step % check_step == 0:
                predicts = nd.argmax(nd.softmax(mt_outputs, axis=-1), axis=-1)
                refer_sample = src.asnumpy().tolist()
                label_sample = label.asnumpy().tolist()
                pred_sample = predicts.asnumpy().tolist()
                logger.info("train sample:")
                logger.info("refer  :{}".format(
                    " ".join([src_vocab.idx_to_token[int(idx)] for idx in refer_sample[0]])).replace(src_vocab.padding_token, ""))
                logger.info("target :{}".format(
                    " ".join([tgt_vocab.idx_to_token[int(idx)] for idx in label_sample[0]])).replace(EOS, "[EOS]").replace(tgt_vocab.padding_token, ""))
                logger.info("predict:{}".format(
                    " ".join([tgt_vocab.idx_to_token[int(idx)] for idx in pred_sample[0]])).replace(EOS, "[EOS]"))

            if global_step and global_step % eval_step == 0:
                dev_bleu_score = eval(src_bert, mt_model, src_vocab,
                                      tgt_vocab, dev_dataiter, logger, ctx=ctx)
                if not os.path.exists(params_save_path_root):
                    os.makedirs(params_save_path_root)
                model_params_file = params_save_path_root + \
                    "src_bert_step_{}.params".format(global_step)
                src_bert.save_parameters(model_params_file)
                logger.info("{} Save Completed.".format(model_params_file))

                model_params_file = params_save_path_root + \
                    "mt_step_{}.params".format(global_step)
                mt_model.save_parameters(model_params_file)
                logger.info("{} Save Completed.".format(model_params_file))
            writer.add_scalar("loss", loss_scalar, global_step)
            global_step += 1


def eval(src_bert, mt_model, src_vocab, tgt_vocab, dev_dataiter, logger, ctx):
    references = []
    hypothesis = []
    score = 0
    chencherry = SmoothingFunction()
    for src, _, label, src_valid_len, label_valid_len in tqdm(dev_dataiter):
        src = src.as_in_context(ctx)
        src_valid_len = src_valid_len.as_in_context(ctx)
        batch_size = src.shape[0]

        src_token_type = nd.zeros_like(src)
        src_bert_outputs = src_bert(src, src_token_type, src_valid_len)

        tgt_sentences = [BOS]
        tgt = tgt_vocab[tgt_sentences]
        tgt = nd.array([tgt], ctx=ctx)
        tgt = nd.broadcast_axes(tgt, axis=0, size=batch_size)

        for n in range(0, args.max_tgt_len):
            mt_outputs = mt_model(src_bert_outputs, src, tgt)
            predicts = nd.argmax(nd.softmax(mt_outputs, axis=-1), axis=-1)
            final_predict = predicts[:, -1:]
            tgt = nd.concat(tgt, final_predict, dim=1)

        label = label.asnumpy().tolist()
        predict_valid_len = nd.sum(nd.not_equal(predicts, tgt_vocab(
            tgt_vocab.padding_token)), axis=-1).asnumpy().tolist()
        predicts = tgt[:, 1:].asnumpy().tolist()
        label_valid_len = label_valid_len.asnumpy().tolist()

        for refer, hypoth, l_v_len, p_v_len in zip(label, predicts, label_valid_len, predict_valid_len):
            l_v_len = int(l_v_len)
            p_v_len = int(p_v_len)
            refer = refer[:l_v_len]
            refer_str = [tgt_vocab.idx_to_token[int(idx)] for idx in refer]
            hypoth_str = [tgt_vocab.idx_to_token[int(idx)] for idx in hypoth]
            hypoth_str_valid = []
            for token in hypoth_str:
                if token == EOS:
                    hypoth_str_valid.append(token)
                    break
                hypoth_str_valid.append(token)
            references.append(refer_str)
            hypothesis.append(hypoth_str_valid)

    for refer, hypoth in zip(references, hypothesis):
        score += sentence_bleu([refer], hypoth,
                               smoothing_function=chencherry.method1)
    logger.info("dev sample:")
    logger.info("refer :{}".format(" ".join(references[0]).replace(
        EOS, "[EOS]").replace(tgt_vocab.padding_token, "")))
    logger.info("hypoth:{}".format(" ".join(hypothesis[0]).replace(
        EOS, "[EOS]")))
    return score / len(references)


def main(args):
    # init some setting
    # config logging
    log_path = os.path.join(args.log_root, '{}.log'.format(args.model_name))
    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)
    logger = config_logger(log_path)

    gpu_idx = args.gpu
    if not gpu_idx:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(gpu_idx - 1)
    logger.info("Using ctx: {}".format(ctx))

    # Loading vocab and model
    src_bert, src_vocab = gluonnlp.model.get_model(args.bert_model,
                                                   dataset_name=args.src_bert_dataset,
                                                   pretrained=True,
                                                   ctx=ctx,
                                                   use_pooler=False,
                                                   use_decoder=False,
                                                   use_classifier=False)
    _, tgt_vocab = gluonnlp.model.get_model(args.bert_model,
                                            dataset_name=args. tgt_bert_dataset,
                                            pretrained=True,
                                            ctx=ctx,
                                            use_pooler=False,
                                            use_decoder=False,
                                            use_classifier=False)

    mt_model = MTModel_Hybird(src_vocab=src_vocab,
                              tgt_vocab=tgt_vocab,
                              embedding_dim=args.mt_emb_dim,
                              model_dim=args.mt_model_dim,
                              head_num=args.mt_head_num,
                              layer_num=args.mt_layer_num,
                              ffn_dim=args.mt_ffn_dim,
                              dropout=args.mt_dropout,
                              att_dropout=args.mt_att_dropout,
                              ffn_dropout=args.mt_ffn_dropout,
                              ctx=ctx)
    logger.info("Model Creating Completed.")

    # init or load params for model
    mt_model.initialize(init.Xavier(), ctx)

    if args.src_bert_load_path:
        src_bert.load_parameters(args.src_bert_load_path, ctx=ctx)
    if args.mt_model_load_path:
        mt_model.load_parameters(args.mt_model_load_path, ctx=ctx)
    logger.info("Parameters Initing and Loading Completed")

    src_bert.hybridize()
    mt_model.hybridize()

    # Loading dataloader
    assiant = DatasetAssiantTransformer(
        src_vocab=src_vocab, tgt_vocab=tgt_vocab, max_src_len=args.max_src_len, max_tgt_len=args.max_tgt_len)
    train_dataset = MTDataset(args.train_data_path)
    eval_dataset = MTDataset(args.eval_data_path)

    train_dataiter = MTDataLoader(train_dataset, batch_size=args.batch_size,
                                  assiant=assiant, shuffle=True).dataiter
    dev_dataiter = MTDataLoader(eval_dataset, batch_size=args.batch_size,
                                assiant=assiant, shuffle=True).dataiter
    logger.info("Data Loading Completed")

    # build trainer
    finetune_trainer = gluon.Trainer(src_bert.collect_params(),
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
        src_bert=src_bert,
        mt_model=mt_model,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        train_dataiter=train_dataiter,
        dev_dataiter=dev_dataiter,
        trainer=trainer,
        finetune_trainer=finetune_trainer,
        epochs=args.epochs,
        loss_func=loss_func,
        ctx=ctx,
        lr=args.train_lr,
        batch_size=args.batch_size,
        params_save_path_root=args.params_save_path_root,
        eval_step=args.eval_step,
        log_step=args.log_step,
        check_step=args.check_step,
        label_smooth=args.label_smooth,
        logger=logger,
        num_train_examples=len(train_dataset),
        warmup_ratio=args.warmup_ratio
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Use Bert's Outputs as Transformer's Encoder Outputs, the Decoder is a Norm Transformer's Decoder.")
    parser.add_argument("--model_name", type=str,
                        default="bert2transformerOnMachineTranslation")
    parser.add_argument("--train_data_path", type=str,
                        default=None, required=True)
    parser.add_argument("--eval_data_path", type=str,
                        default=None, required=True)
    parser.add_argument("--bert_model", type=str,
                        default="bert_12_768_12")
    parser.add_argument("--src_bert_dataset", type=str,
                        default=None, required=True)
    parser.add_argument("--tgt_bert_dataset", type=str,
                        default=None, required=True)
    parser.add_argument("--src_bert_load_path", type=str,
                        default=None)
    parser.add_argument("--mt_model_load_path", type=str,
                        default=None)
    parser.add_argument("--gpu", type=int,
                        default=1, help='which gpu to use for finetuning. CPU is used if set 0.')
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--train_lr", type=float, default=0.0002)
    parser.add_argument("--finetune_lr", type=float, default=2e-5)
    parser.add_argument("--label_smooth", type=bool, default=True)
    parser.add_argument("--batch_size", type=int,
                        default=None, required=True)
    parser.add_argument("--epochs", type=int,
                        default=None, required=True)
    parser.add_argument("--log_root", type=str, default=None, required=True)
    parser.add_argument("--log_step", type=int, default=None, required=True)
    parser.add_argument("--eval_step", type=int, default=None, required=True)
    parser.add_argument("--check_step", type=int, default=None, required=True)
    parser.add_argument("--params_save_path_root",
                        type=str, default="../checkpoints/")
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='ratio of warmup steps that linearly increase learning rate from '
                        '0 to target learning rate. default is 0.1')
    parser.add_argument("--max_src_len", type=int,
                        default=128)
    parser.add_argument("--max_tgt_len", type=int,
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
