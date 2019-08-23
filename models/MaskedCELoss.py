from mxnet.gluon.loss import SoftmaxCrossEntropyLoss
from mxnet import ndarray


class MaskedCELoss(SoftmaxCrossEntropyLoss):
    def __init__(self, axis=-1, sparse_label=True, from_logits=False, weight=None,
                 batch_axis=0, **kwargs):
        super(MaskedCELoss, self).__init__(
            weight, batch_axis, **kwargs)
        self._axis = axis
        self._batch_axis = batch_axis
        self._sparse_label = sparse_label
        self._from_logits = from_logits

    def hybrid_forward(self, F, pred, label, mask=None):
        if not self._from_logits:
            pred = F.log_softmax(pred, self._axis)
        if self._sparse_label:
            loss = -F.pick(pred, label, axis=self._axis, keepdims=True)
        else:
            label = _reshape_like(F, label, pred)
            loss = -F.sum(pred * label, axis=self._axis)
        if mask is not None:
            loss = loss * mask
        return F.sum(loss) / F.sum(mask)


def _reshape_like(F, x, y):
    """Reshapes x to the same shape as y."""
    return x.reshape(y.shape) if F is ndarray else F.reshape_like(x, y)
