def get_data():
    en_sentences = open(
        r"../data/UN(已分词)/(un_parallel_en)origin.en.sentences.train.txt", "r", encoding="utf-8").readlines()
    ch_sentences = open(
        r"../data/UN(已分词)/(un_parallel_ch)origin.ch.sentences.train.txt", "r", encoding="utf-8").readlines()
    en_sentences_save = []
    ch_sentences_save = []
    count = 0
    max_count = 10000
    for en_sent, ch_sent in zip(en_sentences, ch_sentences):
        if len(en_sent.split()) > 20 and len(en_sent.split()) < 30 and len(ch_sent.split()) > 20 and len(ch_sent.split()) < 30:
            en_sentences_save.append(en_sent)
            ch_sentences_save.append(ch_sent)
            count += 1
        if count == max_count:
            break

    open("./data/train.en.sentences", "w",
         encoding="utf-8").writelines(en_sentences_save)
    open("./data/train.ch.sentences", "w",
         encoding="utf-8").writelines(ch_sentences_save)

    print("done!")


if __name__ == "__main__":
    get_data()
