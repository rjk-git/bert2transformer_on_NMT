python translate.py \
 --src_bert_dataset=wiki_cn_cased \
 --tgt_bert_dataset=book_corpus_wiki_en_uncased \
 --bert_model_params_path=../checkpoints/src_bert_step_100.params \
 --mt_model_params_path=../checkpoints/mt_step_100.params \
 --max_src_len=50 \
 --max_tgt_len=50 \
