import dynet as dy
import xnmt


class ResidualSeqTransducer(xnmt.models.SeqTransducer, xnmt.Serializable):
  """
  A sequence transducer that wraps a :class:`xnmt.transducers.base.SeqTransducer` in an additive residual
  connection, and optionally performs some variety of normalization.

  Args:
    child the child transducer to wrap
    layer_norm: whether to perform layer normalization
    dropout: whether to apply residual dropout
  """
  @xnmt.serializable_init
  def __init__(self,
               child: xnmt.models.SeqTransducer,
               input_dim: int,
               layer_norm: bool = False,
               dropout: float = xnmt.default_dropout):
    self.child = child
    self.dropout = dropout
    self.input_dim = input_dim
    self.layer_norm = layer_norm
    if layer_norm:
      model = xnmt.param_manager(self)
      self.ln_g = model.add_parameters(dim=(input_dim,))
      self.ln_b = model.add_parameters(dim=(input_dim,))

  def transduce(self, seq: xnmt.ExpressionSequence) -> xnmt.models.states.EncoderState:
    transduce_result = self.child.transduce(seq)
    seq_tensor = transduce_result.encode_seq
    final_states = transduce_result.encoder_final_states

    if xnmt.is_train() and self.dropout > 0.0:
      seq_tensor = dy.dropout(seq_tensor.as_tensor(), self.dropout) + seq.as_tensor()
    else:
      seq_tensor = seq_tensor.as_tensor() + seq.as_tensor()

    if self.layer_norm:
      d = seq_tensor.dim()
      seq_tensor = dy.reshape(seq_tensor, (d[0][0],), batch_size=d[0][1]*d[1])
      seq_tensor = dy.layer_norm(seq_tensor, self.ln_g, self.ln_b)
      seq_tensor = dy.reshape(seq_tensor, d[0], batch_size=d[1])
    return xnmt.models.states.EncoderState(xnmt.ExpressionSequence(expr_tensor=seq_tensor), final_states)

