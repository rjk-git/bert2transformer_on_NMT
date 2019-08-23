from mxnet import gluon


class MTDataset(gluon.data.Dataset):
    def __init__(self, trans_path, aim_path, **kwargs):
        super(MTDataset, self).__init__(**kwargs)
        self.trans_sentences, self.aim_sentences = self._get_data(trans_path, aim_path)

    def _get_data(self, trans_path, aim_path):
        with open(trans_path, 'r', encoding='utf-8') as fr_trans:
            trans_sentences = [sentence.strip() for sentence in fr_trans.readlines()]
        with open(aim_path, 'r', encoding='utf-8') as fr_aim:
            aim_sentences = [sentence.strip() for sentence in fr_aim.readlines()]
        if len(trans_sentences) != len(aim_sentences):
            assert "lens of trans and aim is not the same!"
        return trans_sentences, aim_sentences

    def __getitem__(self, item):
        return self.trans_sentences[item], self.aim_sentences[item]

    def __len__(self):
        return len(self.trans_sentences)
