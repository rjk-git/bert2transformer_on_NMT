import gluonnlp

from gluonnlp.data import BERTTokenizer, BERTSentenceTransform
from mxnet.gluon.data import DataLoader
from bert_embedding.dataset import BertEmbeddingDataset

from hyperParameters import GetHyperParameters as ghp


def word_piece_tokenizer(sentences):
    ctx = ghp.ctx
    model = 'bert_12_768_12'
    dataset_name = 'book_corpus_wiki_en_uncased'
    max_seq_length = ghp.max_seq_len
    batch_size = 256
    _, vocab = gluonnlp.model.get_model(model,
                                        dataset_name=dataset_name,
                                        pretrained=True, ctx=ctx,
                                        use_pooler=False,
                                        use_decoder=False,
                                        use_classifier=False)
    tokenizer = BERTTokenizer(vocab)

    transform = BERTSentenceTransform(tokenizer=tokenizer,
                                      max_seq_length=max_seq_length,
                                      pair=False)
    dataset = BertEmbeddingDataset(sentences, transform)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
    batches = []
    for token_ids, _, _ in data_loader:
        token_ids = token_ids.as_in_context(ctx)

        for token_id in token_ids.asnumpy():
            batches.append(token_id)

    cut_results = []
    for token_ids in batches:
        tokens = []
        for token_id in token_ids:
            if token_id == 1:
                break
            if token_id in (2, 3):
                continue
            token = vocab.idx_to_token[token_id]
            if token.startswith('##'):
                token = token[2:]
                tokens[-1] += token
            else:  # iv, avg last oov
                tokens.append(token)
        cut_results.append(tokens)
    return cut_results
