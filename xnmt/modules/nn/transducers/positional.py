
import dynet as dy

import xnmt


# Note: alternatively, this could wrap "PositionEmbedder", but it seems to me
#       that PositionEmbedder is probably not necessary in the first place, so
#       it probably makes more sense to have this as a SeqTransducer that
#       adds positional embeddings to an input
import xnmt.models.templates


class PositionalSeqTransducer(xnmt.models.SeqTransducer, xnmt.Serializable):

  yaml_tag = '!PositionalSeqTransducer'

  @xnmt.serializable_init
  def __init__(self,
               max_pos: int,
               op: str = 'sum',
               emb_type: str = 'param',
               input_dim: input = xnmt.default_layer_dim,
               dropout: float = xnmt.default_dropout,
               param_init: xnmt.models.templates.ParamInitializer = xnmt.default_param_init):
    """
    max_pos: largest embedded position
    op: how to combine positional encodings with the original encodings, can be "sum" or "concat"
    type: what type of embddings to use, "param"=parameterized (others, such as the trigonometric embeddings are todo)
    input_dim: embedding size
    dropout: apply dropout to output of this transducer
    param_init: how to initialize embedding matrix
    """
    self.max_pos = max_pos
    self.input_dim = input_dim
    self.dropout = dropout
    self.op = op
    self.emb_type = emb_type
    dim = (self.input_dim, max_pos)
    param_collection = xnmt.param_manager(self)
    self.embedder = param_collection.add_parameters(dim, init=param_init.initializer(dim, is_lookup=True))

  def transduce(self, src: xnmt.ExpressionSequence) -> xnmt.models.states.EncoderState:
    sent_len = len(src)
    embeddings = dy.strided_select(dy.parameter(self.embedder), [1,1], [0,0], [self.input_dim, sent_len])

    if self.op == 'sum':
      output = embeddings + src.as_tensor()
    elif self.op == 'concat':
      output = dy.concatenate([embeddings, src.as_tensor()])
    else:
      raise ValueError(f'Illegal op {op} in PositionalTransducer (options are "sum"/"concat")')

    if xnmt.is_train() and self.dropout > 0.0:
      output = dy.dropout(output, self.dropout)

    output_seq = xnmt.ExpressionSequence(expr_tensor=output, mask=src.mask)
    return xnmt.models.states.EncoderState(output_seq, [xnmt.models.states.FinalTransducerState(output_seq[-1])])
