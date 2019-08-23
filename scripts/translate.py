import sys
import re
from pypinyin import pinyin
sys.path.append("..")
import argparse
from models.MTModel_Hybird import Transformer as MTModel_Hybird
from gluonnlp.data import BERTTokenizer
import mxnet as mx
import gluonnlp
from mxnet import nd
BOS = "[unused16]"
EOS = "[unused17]"


def translate(args):
    gpu_idx = args.gpu
    if not gpu_idx:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(gpu_idx - 1)
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

    en_bert.load_parameters(args.en_bert_model_params_path, ctx=ctx)
    mt_model.load_parameters(args.mt_model_params_path, ctx=ctx)

    en_bert_tokenzier = BERTTokenizer(en_vocab)
    ch_bert_tokenzier = BERTTokenizer(ch_vocab)

    while True:
        trans = input("input:")

        trans = en_bert_tokenzier(trans)
        trans = [en_vocab.cls_token] + \
            trans + [en_vocab.sep_token]

        trans_valid_len = len(trans)

        if args.max_en_len and len(trans) > args.max_en_len:
            trans = trans[0:args.max_en_len]

        aim = [BOS]

        trans = en_vocab[trans]
        aim = ch_vocab[aim]

        aim = nd.array([aim], ctx=ctx)

        trans = nd.array([trans], ctx=ctx)
        trans_valid_len = nd.array([trans_valid_len], ctx=ctx)
        trans_token_types = nd.zeros_like(trans)

        batch_size = 1
        beam_size = 6

        en_bert_outputs = en_bert(trans, trans_token_types, trans_valid_len)
        mt_outputs = mt_model(en_bert_outputs, trans, aim)

        en_bert_outputs = nd.broadcast_axes(en_bert_outputs, axis=0, size=beam_size)
        trans = nd.broadcast_axes(trans, axis=0, size=beam_size)
        targets = None
        for n in range(0, args.max_ch_len):
            aim, targets = beam_search(
                mt_outputs[:, n, :], targets=targets, max_seq_len=args.max_ch_len, ctx=ctx, beam_width=beam_size)
            mt_outputs = mt_model(en_bert_outputs, trans, aim)

        predict = aim.asnumpy().tolist()
        predict_strs = []
        for pred in predict:
            predict_token = [ch_vocab.idx_to_token[int(idx)] for idx in pred]
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
    predicts = nd.topk(nd.softmax(outputs, axis=-1), axis=-1, k=beam_width, ret_typ='both')

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
                        default=10)
    parser.add_argument("--valid_ratio", type=int,
                        default=0.0001)
    parser.add_argument("--log_root", type=str, default="../logs/")
    parser.add_argument("--log_step", type=int, default=10)
    parser.add_argument("--eval_step", type=int, default=500)
    parser.add_argument("--params_save_step", type=int, default=5000)
    parser.add_argument("--params_save_path_root", type=str, default="../parameters/")
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

    translate(args)
