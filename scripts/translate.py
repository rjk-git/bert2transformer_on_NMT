import argparse
import re
import sys
sys.path.append("..")

import gluonnlp
import mxnet as mx
from gluonnlp.data import BERTTokenizer
from models.MTModel_Hybird import Transformer as MTModel_Hybird
from mxnet import nd

BOS = "[CLS]"
EOS = "[SEP]"


def translate(args):
    gpu_idx = args.gpu
    if not gpu_idx:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(gpu_idx - 1)
    src_bert, src_vocab = gluonnlp.model.get_model(args.bert_model,
                                                   dataset_name=args.src_bert_dataset,
                                                   pretrained=True,
                                                   ctx=ctx,
                                                   use_pooler=False,
                                                   use_decoder=False,
                                                   use_classifier=False)
    _, tgt_vocab = gluonnlp.model.get_model(args.bert_model,
                                            dataset_name=args.tgt_bert_dataset,
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

    src_bert.load_parameters(args.bert_model_params_path, ctx=ctx)
    mt_model.load_parameters(args.mt_model_params_path, ctx=ctx)

    src_bert_tokenzier = BERTTokenizer(src_vocab)
    tgt_bert_tokenzier = BERTTokenizer(tgt_vocab)

    while True:
        src = input("input:")

        src = src_bert_tokenzier(src)
        src = [src_vocab.cls_token] + \
            src + [src_vocab.sep_token]

        src_valid_len = len(src)

        if args.max_src_len and len(src) > args.max_src_len:
            src = src[0:args.max_src_len]

        tgt = [BOS]

        src = src_vocab[src]
        tgt = tgt_vocab[tgt]

        tgt = nd.array([tgt], ctx=ctx)

        src = nd.array([src], ctx=ctx)
        src_valid_len = nd.array([src_valid_len], ctx=ctx)
        src_token_types = nd.zeros_like(src)

        beam_size = 6

        src_bert_outputs = src_bert(src, src_token_types, src_valid_len)
        mt_outputs = mt_model(src_bert_outputs, src, tgt)

        src_bert_outputs = nd.broadcast_axes(
            src_bert_outputs, axis=0, size=beam_size)
        src = nd.broadcast_axes(src, axis=0, size=beam_size)
        targets = None
        for n in range(0, args.max_tgt_len):
            tgt, targets = beam_search(
                mt_outputs[:, n, :], targets=targets, max_seq_len=args.max_tgt_len, ctx=ctx, beam_width=beam_size)
            mt_outputs = mt_model(src_bert_outputs, src, tgt)

        predict = tgt.asnumpy().tolist()
        predict_strs = []
        for pred in predict:
            predict_token = [tgt_vocab.idx_to_token[int(idx)] for idx in pred]
            predict_str = ""
            sub_token = []
            for token in predict_token:
                # if token in ["[CLS]", EOS, "[SEP]"]:
                #     continue
                if len(sub_token) == 0:
                    sub_token.append(token)
                elif token[:2] != "##" and len(sub_token) != 0:
                    predict_str += "".join(sub_token) + " "
                    sub_token = []
                    sub_token.append(token)
                else:
                    if token[:2] == "##":
                        token = token.replace("##", "")
                    sub_token.append(token)
                if token == EOS:
                    if len(sub_token) != 0:
                        predict_str += "".join(sub_token) + " "
                    break
            predict_strs.append(predict_str.replace(
                "[SEP]", "").replace("[CLS]", "").replace(EOS, ""))
        for predict_str in predict_strs:
            print(predict_str)


def beam_search(outputs, ctx, targets, max_seq_len, beam_width):
    predicts = nd.topk(nd.softmax(outputs, axis=-1),
                       axis=-1, k=beam_width, ret_typ='both')

    if not targets:
        targets = {}
        beam_result_idxs = []
        beam_result_score = []
        count = 0
        for score, idx in zip(predicts[0][0], predicts[1][0]):
            idx = [2] + [int(idx.asscalar())]
            beam_result_idxs.append(idx)
            beam_result_score.append(score)
            targets.update(
                {"beam_{}".format(count): {"idx": idx, "score": score}})
            count += 1

        result = []
        for idx in beam_result_idxs:
            idx = idx[:max_seq_len] + \
                [0] * (max_seq_len - len(idx))
            result.append(idx)
        return nd.array(result, ctx=ctx), targets

    else:
        beam_idxs = []
        beam_score = []
        for scores, idxs, target in zip(predicts[0], predicts[1], targets.values()):
            last_score = target["score"]
            last_idxs = target["idx"]
            max_score = 0
            max_score_idx = []
            for score, idx in zip(scores, idxs):
                if last_score + score > max_score:
                    max_score = last_score + score
                    idx = int(idx.asscalar())
                    max_score_idx = last_idxs[:] + [idx]

            beam_idxs.append(max_score_idx)
            beam_score.append(max_score)

        beam_score, beam_idxs = (list(t)
                                 for t in zip(*sorted(zip(beam_score, beam_idxs), reverse=True)))

        targets = {}
        count = 0
        for idx, score in zip(beam_idxs, beam_score):
            targets.update(
                {"beam_{}".format(count): {"idx": idx, "score": score}})
            count += 1

        result = []
        for idx in beam_idxs:
            idx = idx[:max_seq_len] + \
                [0] * (max_seq_len - len(idx))
            result.append(idx)
        return nd.array(result, ctx=ctx), targets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str,
                        default="Bert2transformer_translate")
    parser.add_argument("--bert_model", type=str,
                        default="bert_12_768_12")
    parser.add_argument("--src_bert_dataset", type=str,
                        default=None, required=True)
    parser.add_argument("--tgt_bert_dataset", type=str,
                        default=None, required=True)
    parser.add_argument("--bert_model_params_path", type=str,
                        default=None, required=True)
    parser.add_argument("--mt_model_params_path", type=str,
                        default=None, required=True)
    parser.add_argument("--gpu", type=int,
                        default=1, help='which gpu to use for finetuning. CPU is used if set 0.')
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

    translate(args)
