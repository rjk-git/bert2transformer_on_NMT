from mxnet import gluon


class MTDataset(gluon.data.Dataset):
    def __init__(self, data_path, **kwargs):
        super(MTDataset, self).__init__(**kwargs)
        self.src_sentences, self.tgt_sentences = self._get_data(data_path)

    def _get_data(self, data_path, sep="\t"):
        src_sentences = []
        tgt_sentences = []
        with open(data_path, 'r', encoding='utf-8') as fr_trans:
            lines = [line.strip()
                     for line in fr_trans.readlines()]
        for line in lines:
            src, tgt = line.split(sep)
            # tgt, src = line.split(sep)
            src_sentences.append(src)
            tgt_sentences.append(tgt)
        if len(src_sentences) != len(tgt_sentences):
            assert "lens of SRC and TGT is not the same!"
        return src_sentences, tgt_sentences

    def __getitem__(self, item):
        return self.src_sentences[item], self.tgt_sentences[item]

    def __len__(self):
        return len(self.src_sentences)
