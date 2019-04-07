import numpy as np

from mxnet import nd

from scripts.hyperParameters import GetHyperParameters as ghp


def getMask(q_seq, k_seq):
    # q_seq shape : (batch_size, q_seq_len)
    # k_seq shape : (batch_size, k_seq_len)
    q_len = q_seq.shape[1]
    pad_mask = nd.not_equal(k_seq, 0)
    pad_mask = nd.expand_dims(pad_mask, axis=1)
    pad_mask = nd.broadcast_axes(pad_mask, axis=1, size=q_len)

    return pad_mask

def getSelfMask(q_seq):
    batch_size, seq_len = q_seq.shape
    mask_matrix = np.ones(shape=(seq_len, seq_len), dtype=np.float)
    mask = np.tril(mask_matrix, k=0)
    mask = nd.expand_dims(nd.array(mask, ctx=ghp.ctx), axis=0)
    mask = nd.broadcast_axes(mask, axis=0, size=batch_size)
    return mask

if __name__ == '__main__':
    mask = getMask(nd.array([[1,2,0],[1,0,0]]), nd.array([[1,2,3],[5,0,0]]))
    print(mask)
    mask = getSelfMask(nd.array([[1,2,0], [2,0,0], [0,0,0]]))
    print(mask)

    score = nd.array([[5, 6, 0], [5, 10, 0]])
    pading = nd.ones_like(score) * 0.1
    score = nd.where(nd.equal(mask[0], 0), pading, score)
    print(score)