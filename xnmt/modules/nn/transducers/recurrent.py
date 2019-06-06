import dynet as dy
import xnmt
import numpy as np
import xnmt.models as models

from typing import Optional, Sequence, List

class MaskedRNNState(models.UniDirectionalState):
  
  def __init__(self, state: dy.RNNState):
    self.state = state
    
  def add_input(self, word: dy.Expression, mask: Optional[xnmt.Mask] = None, position: int = 0):
    next_state = self.state.add_input(word)
    if mask is not None and len(self.state.s()) > 0:
      s = [mask.cmult_by_timestep_expr(x, position) + mask.cmult_by_timestep_expr(y, position, inverse=True) \
           for x, y in zip(self.state.s(), next_state.s())]
      next_state = next_state.set_s(s)
    return MaskedRNNState(next_state)
  
  def output(self):
    return self.state.output()
  
  def context(self):
    raise NotImplementedError()
  
  def position(self):
    raise NotImplementedError()


class UniLSTMSeqTransducer(xnmt.models.UniDiSeqTransducer, xnmt.Serializable):
  yaml_tag = "!UniLSTMSeqTransducer"
  """
  This implements a single LSTM layer based on the memory-friendly dedicated DyNet nodes.
  It works similar to DyNet's CompactVanillaLSTMBuilder, but in addition supports
  taking multiple inputs that are concatenated on-the-fly.

  Args:
    layers (int): number of layers
    input_dim (int): input dimension
    hidden_dim (int): hidden dimension
    dropout (float): dropout probability
    weightnoise_std (float): weight noise standard deviation
    param_init (ParamInitializer): how to initialize weight matrices
    bias_init (ParamInitializer): how to initialize bias vectors
    yaml_path (str):
    decoder_input_dim (int): input dimension of the decoder; if ``yaml_path`` contains 'decoder' and ``decoder_input_feeding`` is True, this will be added to ``input_dim``
    decoder_input_feeding (bool): whether this transducer is part of an input-feeding decoder; cf. ``decoder_input_dim``
  """
  @xnmt.serializable_init
  def __init__(self,
               layers: int = 1,
               input_dim: int = xnmt.default_layer_dim,
               hidden_dim: int = xnmt.default_layer_dim,
               dropout: float = xnmt.default_dropout,
               weightnoise_std: float = xnmt.default_weight_noise,
               param_init: xnmt.ParamInitializer = xnmt.default_param_init,
               bias_init: xnmt.ParamInitializer = xnmt.default_bias_init,
               yaml_path: xnmt.Path = xnmt.Path(),
               decoder_input_dim: Optional[int] = xnmt.default_layer_dim_optional,
               decoder_input_feeding: bool = True) -> None:
    model = xnmt.param_manager(self)
    if yaml_path is not None and "decoder" in yaml_path:
      if decoder_input_feeding:
        input_dim += decoder_input_dim
    
    self.builder = dy.CompactVanillaLSTMBuilder(layers, input_dim, hidden_dim, model)
    self.dropout = dropout
    self.builder.set_weightnoise(weightnoise_std)
    self.total_input_dim = input_dim
    
  def initial_state(self, init: List[dy.Expression] = None) -> MaskedRNNState:
    if not xnmt.globals.is_train() and self.dropout > 0.0:
      self.builder.set_dropouts(self.dropout, self.dropout)
    else:
      self.builder.disable_dropout()
    ret = self.builder.initial_state()
    if init is not None:
      ret = ret.set_s(init)
    return MaskedRNNState(ret)
    
  def add_input(self, prev_state: MaskedRNNState, x: dy.Expression, mask: Optional[xnmt.Mask], position: int = 0) \
      -> models.states.UniDirectionalState:
    return prev_state.add_input(x, mask, position)
    

  def transduce(self, expr_seq: xnmt.ExpressionSequence) -> models.EncoderState:
    """
    transduce the sequence, applying masks if given (masked timesteps simply copy previous h / c)

    Args:
      expr_seq: expression sequence or list of expression sequences (where each inner list will be concatenated)
    Returns:
      expression sequence
    """
    state = self.initial_state()
    out_expr = []
    for i in range(len(expr_seq)):
      state = self.add_input(state, expr_seq[i], expr_seq.mask, i)
      out_expr.append(state.output())

    out_expr = xnmt.ExpressionSequence(expr_list=out_expr, mask=expr_seq.mask)
    state_s = state.state.s()
    offset = len(state_s) // 2
    final_states = [models.states.FinalTransducerState(state_s[offset+i], state_s[i]) for i in range(offset)]
    return models.EncoderState(out_expr, final_states)


class BiLSTMSeqTransducer(models.BidiSeqTransducer, xnmt.Serializable):
  yaml_tag = "!BiLSTMSeqTransducer"
  """
  This implements a bidirectional LSTM and requires about 8.5% less memory per timestep
  than DyNet's CompactVanillaLSTMBuilder due to avoiding concat operations.
  It uses 2 :class:`xnmt.lstm.UniLSTMSeqTransducer` objects in each layer.

  Args:
    layers (int): number of layers
    input_dim (int): input dimension
    hidden_dim (int): hidden dimension
    dropout (float): dropout probability
    weightnoise_std (float): weight noise standard deviation
    param_init: a :class:`xnmt.param_init.ParamInitializer` or list of :class:`xnmt.param_init.ParamInitializer` objects
                specifying how to initialize weight matrices. If a list is given, each entry denotes one layer.
    bias_init: a :class:`xnmt.param_init.ParamInitializer` or list of :class:`xnmt.param_init.ParamInitializer` objects
               specifying how to initialize bias vectors. If a list is given, each entry denotes one layer.
    forward_layers: set automatically
    backward_layers: set automatically
  """
  @xnmt.serializable_init
  def __init__(self,
               layers: int = 1,
               input_dim: int = xnmt.default_layer_dim,
               hidden_dim: int = xnmt.default_layer_dim,
               dropout: float = xnmt.default_dropout,
               weightnoise_std: float = xnmt.default_weight_noise,
               param_init: xnmt.ParamInitializer = xnmt.default_param_init,
               bias_init: xnmt.ParamInitializer = xnmt.default_bias_init,
               forward_layers : Optional[Sequence[UniLSTMSeqTransducer]] = None,
               backward_layers: Optional[Sequence[UniLSTMSeqTransducer]] = None) -> None:
    self.hidden_dim = hidden_dim
    assert hidden_dim % 2 == 0
    self.forward_layers = self.add_serializable_component("forward_layers", forward_layers, lambda:
      UniLSTMSeqTransducer(input_dim=input_dim,
                           hidden_dim=hidden_dim // 2, dropout=dropout,
                           weightnoise_std=weightnoise_std,
                           param_init=param_init,
                           bias_init=bias_init,
                           layers = layers))
    self.backward_layers = self.add_serializable_component("backward_layers", backward_layers, lambda:
      UniLSTMSeqTransducer(input_dim=input_dim,
                           hidden_dim=hidden_dim // 2,
                           dropout=dropout, weightnoise_std=weightnoise_std,
                           param_init=param_init,
                           bias_init=bias_init,
                           layers = layers))


  def transduce(self, es: xnmt.ExpressionSequence) -> models.EncoderState:
     # first layer
    fwd_encode = self.forward_layers.transduce(es)
    batch_size = es[0].dim()[1]
    
    if es.mask is not None:
      offset = [np.count_nonzero(x) for x in es.mask.np_arr]
      el = [
        dy.concatenate_to_batch(
          [dy.pick_batch_elem(es[max(-offset[j]-i-1, -len(es))], j) for j in range(batch_size)]
        ) for i in range(len(es))
      ]
      bwd_es = xnmt.ExpressionSequence(expr_list=el, mask=es.mask)
    else:
      bwd_es = xnmt.ExpressionSequence(expr_list=list(reversed(es.expr_list)))
      
    bwd_encode = self.backward_layers.transduce(bwd_es)
    fwd_es = fwd_encode.encode_seq
    bwd_es = bwd_encode.encode_seq
    fwd_final_states = fwd_encode.encoder_final_states
    bwd_final_states = bwd_encode.encoder_final_states
    
    expr_list = [dy.concatenate([fw, bw]) for fw, bw in zip(fwd_es, bwd_es)]
    final_states = [models.FinalTransducerState(main_expr=dy.concatenate([fwd_final_state.main_expr(),
                                                                          bwd_final_state.main_expr()]),
                                                cell_expr=dy.concatenate([fwd_final_state.cell_expr(),
                                                                          bwd_final_state.cell_expr()])) \
                    for fwd_final_state, bwd_final_state in zip(fwd_final_states, bwd_final_states)]

    expr_seq = xnmt.ExpressionSequence(expr_list=expr_list, mask=es.mask)
    return models.EncoderState(expr_seq, final_states)

