def position_encoding(self, zh_idx):
    seq_lens = self.get_seq_len(zh_idx)
    # seq_lens shape : (batch_size, 1)
    position_enc = np.arange(self.zh_max_seq_len).reshape((-1, 1)) \
                   / (np.power(10000, (2. / self.model_dims) * np.arange(self.model_dims).reshape((1, -1))))
    position_enc = nd.array(position_enc)
    position_enc[:, 0::2] = nd.sin(position_enc[:, 0::2])  # dim 2i
    position_enc[:, 1::2] = nd.cos(position_enc[:, 1::2])  # dim 2i+1
    pad_row = nd.array([[0] * self.model_dims])
    # position_enc shape : (zh_max_seq_len + 1, model_dims)
    position_enc = nd.concat(position_enc, pad_row, dim=0)
    # position_enc shape : (1, zh_max_seq_len + 1, model_dims)
    position_enc = nd.expand_dims(position_enc, 0)
    # position_enc shape : (batch_size, zh_max_seq_len + 1, model_dims)
    position_enc = nd.broadcast_axis(position_enc, axis=0, size=len(seq_lens))
    # input_pos_one_hot shape : (batch_size, zh_max_seq_len, zh_max_seq_len + 1)
    input_pos_one_hot = nd.array([list(
        list(0 if __ is not _ else 1 for __ in range(self.zh_max_seq_len + 1)) for _ in range(seq_len)) + list(
        list(0 if __ is not self.zh_max_seq_len else 1 for __ in range(self.zh_max_seq_len + 1)) for _ in
        range(self.zh_max_seq_len - seq_len)) for seq_len in
                                  seq_lens])
    # position_output shape : (batch_size, zh_max_seq_len, model_dims)
    position_output = nd.batch_dot(input_pos_one_hot, position_enc)
    return position_output


def get_seq_len(self, zh_idx):
    seq_lens = []
    count = 0
    for input in zh_idx:
        for x in input:
            if x != 0:
                count += 1
        seq_lens.append(count)
        count = 0
    return seq_lens