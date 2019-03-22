from mxnet import nd
from prepo import load_zh_vocab


def translate(transformer_model, seq):
    zh2idx, idx2zh = load_zh_vocab()
    en_seq_emb, en_seq_len = en_embedding(seq)
    en_seq_emb = nd.array(en_seq_emb)

    en_seq_idx = [[1]*en_seq_len]
    en_seq_idx = nd.array(en_seq_idx)

    dec_begin_input = nd.array([zh2idx["<bos>"]])
    dec_input = nd.expand_dims(dec_begin_input, axis=0)

    zh_max_seq_len = 30
    zh_seq_len = 1
    predict_token = []
    dec_input_list = [zh2idx["<bos>"]]
    while True:
        output, _, _ = transformer_model(en_seq_emb, en_seq_idx, dec_input)
        output = nd.softmax(output, axis=-1)
        predict_idx = nd.argmax(output, axis=-1)
        idx = int(predict_idx[0][zh_seq_len-1].asscalar())
        predict_token.append(idx2zh[idx])
        dec_input_list.append(idx)
        dec_input = nd.array([dec_input_list])
        zh_seq_len += 1
        if zh_seq_len == zh_max_seq_len:
            predict_token.append("<eos>")
            break
        elif predict_token[-1] == "<eos>":
            break
    print("test_result:", "".join(predict_token))
def en_embedding(en_seq):
    en_seq_emb, en_seq_len = bert(en_seq)
    return en_seq_emb, en_seq_len