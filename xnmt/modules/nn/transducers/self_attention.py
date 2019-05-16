import dynet as dy
import numpy as np
import xnmt


class MultiHeadAttentionSeqTransducer(xnmt.models.SeqTransducer, xnmt.Serializable):
  """
  This implements the Multi-headed attention layer of "Attention is All You Need":
  https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf

  Args:
    input_dim: size of inputs
    dropout: dropout to apply to attention matrix
    param_init: how to initialize param matrices
    bias_init: how to initialize bias params
    num_heads: number of attention heads
  """
  @xnmt.serializable_init
  def __init__(self,
               input_dim: int = xnmt.default_layer_dim,
               dropout: float = xnmt.default_dropout,
               param_init: xnmt.models.templates.ParamInitializer = xnmt.default_param_init,
               bias_init: xnmt.models.templates.ParamInitializer = xnmt.default_bias_init,
               num_heads: int = 8):
    assert input_dim % num_heads == 0

    self.dropout = dropout

    param_collection = xnmt.param_manager(self)

    self.input_dim = input_dim
    self.num_heads = num_heads
    self.head_dim = input_dim // num_heads

    self.pWq, self.pWk, self.pWv, self.pWo = [param_collection.add_parameters(dim=(input_dim, input_dim), init=param_init.initializer((input_dim, input_dim))) for _ in range(4)]
    self.pbq, self.pbk, self.pbv, self.pbo = [param_collection.add_parameters(dim=(1, input_dim), init=bias_init.initializer((1, input_dim,))) for _ in range(4)]

  def transduce(self, expr_seq: xnmt.ExpressionSequence) -> xnmt.models.states.EncoderState:
    """
    transduce the sequence

    Args:
      expr_seq: expression sequence or list of expression sequences (where each inner list will be concatenated)
    Returns:
      expression sequence
    """

    Wq, Wk, Wv, Wo = self.pWq, self.pWk, self.pWv, self.pWo
    bq, bk, bv, bo = self.pbq, self.pbk, self.pbv, self.pbo

    # Start with a [(length, model_size) x batch] tensor
    x = expr_seq.as_transposed_tensor()
    x_len = x.dim()[0][0]
    x_batch = x.dim()[1]
    # Get the query key and value vectors
    # TODO: do we need bias broadcasting in DyNet?
    # q = dy.affine_transform([bq, x, Wq])
    # k = dy.affine_transform([bk, x, Wk])
    # v = dy.affine_transform([bv, x, Wv])
    q = bq + x * Wq
    k = bk + x * Wk
    v = bv + x * Wv

    # Split to batches [(length, head_dim) x batch * num_heads] tensor
    q, k, v = [dy.reshape(x, (x_len, self.head_dim), batch_size=x_batch * self.num_heads) for x in (q,k,v)]

    # Do scaled dot product [(length, length) x batch * num_heads], rows are queries, columns are keys
    attn_score = q * dy.transpose(k) / np.sqrt(self.head_dim)
    if expr_seq.mask is not None:
      mask = dy.inputTensor(np.repeat(expr_seq.mask.np_arr, self.num_heads, axis=0).transpose(), batched=True) * -xnmt.globals.INF
      attn_score = attn_score + mask
    attn_prob = dy.softmax(attn_score, d=1)
    if xnmt.is_train() and self.dropout > 0.0:
      attn_prob = dy.dropout(attn_prob, self.dropout)
    # Reduce using attention and resize to match [(length, model_size) x batch]
    o = dy.reshape(attn_prob * v, (x_len, self.input_dim), batch_size=x_batch)
    # Final transformation
    # o = dy.affine_transform([bo, attn_prob * v, Wo])
    o = bo + o * Wo

    expr_seq = xnmt.ExpressionSequence(expr_transposed_tensor=o, mask=expr_seq.mask)
    final_states = [xnmt.models.states.FinalTransducerState(expr_seq[-1], None)]

    return xnmt.models.states.EncoderState(expr_seq, final_states)


