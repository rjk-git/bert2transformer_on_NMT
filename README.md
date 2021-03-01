# bert_transformer_for_machine_translation
BERT2Transformer(Decoder)
Deep learning framework using Mxnet (gluon, gluonnlp)

# Prepare
1.The arguments require a training file and an evaluation file, with each line of each file being a parallel corpus pair, separated by "\t" by default.
___________
2.There is no need for operations such as splitting words and building dictionaries. The encoder of the model is a pre-trained BERT, and it is good to use its corresponding Tokenizer. The decoder also does not need to build its own dictionary, and uses the dictionary of BERT of a certain language. The dictionaries of the encoder and decoder are set in "--src_bert_dataset" and "--tgt_bert_dataset", and the specific BERT versions that can be used are listed below.
### The supported bert datasets are:
    'book_corpus_wiki_en_cased',
    'book_corpus_wiki_en_uncased',
    'wiki_cn_cased',
    'openwebtext_book_corpus_wiki_en_uncased',
    'wiki_multilingual_uncased',
    'wiki_multilingual_cased',

___________
3.You simply need to check whether the source and target languages to be used have their corresponding BERT versions. If it is OK, you just need to prepare the training parallel corpus and set it up in the arguments.

## Environment
Use requirement.txt to install the required packages, the first time you use it, you will automatically download the corresponding BERT pre-training parameters.

# Train
Set your arguments in train.py, then 'python train.py'

# Translate
If you need to use the trained model for translation, please use translate.py and set the trained model parameter address in the arguments.