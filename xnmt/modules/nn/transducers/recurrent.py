import dynet as dy
import xnmt
import collections.abc as abc

import xnmt.models as models

from typing import Optional, Sequence


class UniLSTMState(models.UniDirectionalState):
  """
  State object for UniLSTMSeqTransducer.
  """
  def __init__(self,
               network: 'UniLSTMSeqTransducer',
               prev: Optional['UniLSTMState'] = None,
               c: Sequence[dy.Expression] = None,
               h: Sequence[dy.Expression] = None,
               dropout_mask = None,
               position: int = 0,
               init: Optional[Sequence[dy.Expression]] = None):
    self._network = network
    if init is not None:
      self.set_s(init)

    self._c = c
    self._h = h
    self._prev = prev
    self._dropout_mask = dropout_mask
    self._position = position

  def add_input(self, x: dy.Expression, mask: Optional[xnmt.Mask] = None) -> models.UniDirectionalState:
    network = self._network
    weight_noise = self._network.weightnoise_std if xnmt.is_train() else 0
    batch_size = x[0].dim()[1]

    if self._dropout_mask is None:
      self._dropout_mask = self.calc_dropout_mask(batch_size)
    dropout_mask_x, dropout_mask_h = self._dropout_mask
    new_c, new_h = [], []
    for i in range(self._network.num_layers):
      if self._c is None:
        self._c = [dy.zeros(dim=(network.hidden_dim,), batch_size=batch_size) for _ in range(network.num_layers)]
        self._h = [dy.zeros(dim=(network.hidden_dim,), batch_size=batch_size) for _ in range(network.num_layers)]
      if dropout_mask_x is not None and dropout_mask_h is not None:
        # apply dropout according to https://arxiv.org/abs/1512.05287 (tied weights)
        gates = dy.vanilla_lstm_gates_dropout_concat([x],
                                                     self._h[i],
                                                     network.Wx[i],
                                                     network.Wh[i],
                                                     dy.pick(network.b[i], 0),
                                                     dropout_mask_x[i],
                                                     dropout_mask_h[i],
                                                     weight_noise)
      else:
        gates = dy.vanilla_lstm_gates_concat([x],
                                             self._h[i],
                                             network.Wx[i],
                                             network.Wh[i],
                                             dy.pick(network.b[i]),
                                             weight_noise)
      c_t = dy.vanilla_lstm_c(self._c[i], gates)
      h_t = dy.vanilla_lstm_h(c_t, gates)
      if mask is not None:
        c_t = mask.cmult_by_timestep_expr(c_t, self._position,True) + \
              mask.cmult_by_timestep_expr(self._c[i], self._position, False)
        h_t = mask.cmult_by_timestep_expr(h_t, self._position,True) + \
              mask.cmult_by_timestep_expr(self._h[i], self._position, False)
      new_c.append(c_t)
      new_h.append(h_t)
      x = new_h[-1]

    return UniLSTMState(self._network, prev=self, c=new_c, h=new_h, dropout_mask=self._dropout_mask, position=self._position+1)

  def b(self) -> 'UniLSTMSeqTransducer':
    return self._network

  def h(self) -> Sequence[dy.Expression]:
    return self._h

  def c(self) -> Sequence[dy.Expression]:
    return self._c

  def s(self) -> Sequence[dy.Expression]:
    return self._c + self._h

  def prev(self) -> 'UniLSTMState':
    return self._prev

  def calc_dropout_mask(self, batch_size):
    network = self._network
    if network.dropout_rate > 0.0 and xnmt.is_train():
      retention_rate = 1.0 - network.dropout_rate
      scale = 1.0 / retention_rate
      dropout_mask = lambda x: dy.random_bernoulli((x,), retention_rate, scale, batch_size=batch_size)
      dropout_mask_x = [dropout_mask(network.total_input_dim if i == 0 else network.hidden_dim) for i in range(network.num_layers)]
      dropout_mask_h = [dropout_mask(network.hidden_dim) for _ in range(network.num_layers)]
    else:
      dropout_mask_x = None
      dropout_mask_h = None
    return dropout_mask_x, dropout_mask_h

  def set_h(self, es: Optional[Sequence[dy.Expression]] = None) -> 'UniLSTMState':
    if es is not None:
      assert len(es) == self._network.num_layers
      self._h = tuple(es)
    return self

  def set_s(self, es: Optional[Sequence[dy.Expression]] = None) -> 'UniLSTMState':
    if es is not None:
      assert len(es) == 2 * self._network.num_layers
      self._c = tuple(es[:self._network.num_layers])
      self._h = tuple(es[self._network.num_layers:])
    return self

  def output(self) -> dy.Expression:
    return self._h[-1]

  def __getitem__(self, item):
    return UniLSTMState(network=self._network,
                        prev=self._prev,
                        c=[ci[item] for ci in self._c],
                        h=[hi[item] for hi in self._h],
                        dropout_mask=[maski[item] for maski in self._dropout_mask],
                        position=self._position)


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
    self.num_layers = layers
    model = xnmt.param_manager(self)
    self.hidden_dim = hidden_dim
    self.dropout_rate = dropout
    self.weightnoise_std = weightnoise_std
    self.input_dim = input_dim
    self.total_input_dim = input_dim
    if yaml_path is not None and "decoder" in yaml_path:
      if decoder_input_feeding:
        self.total_input_dim += decoder_input_dim

    if not isinstance(param_init, abc.Sequence):
      param_init = [param_init] * layers
    if not isinstance(bias_init, abc.Sequence):
      bias_init = [bias_init] * layers

    # [i; f; o; g]
    init = lambda x, y, i, z: model.add_parameters(dim=(x, y), init=z[i].initializer((x, y), num_shared=4))
    pinit = lambda x, y, i: init(x, y, i, param_init)
    binit = lambda x, y, i: init(x, y, i, bias_init)
    self.Wx = [pinit(hidden_dim*4, self.total_input_dim, 0)] + [pinit(hidden_dim*4, hidden_dim, i) for i in range(1, layers)]
    self.Wh = [pinit(hidden_dim*4, hidden_dim, i) for i in range(layers)]
    self.b  = [binit(1, hidden_dim*4, i) for i in range(layers)]


  def initial_state(self, init=None) -> UniLSTMState:
    return UniLSTMState(self, init=init)

  def add_input(self, prev_state: UniLSTMState, x: dy.Expression, mask: Optional[xnmt.Mask]) \
      -> models.states.UniDirectionalState:
    return prev_state.add_input(x, mask)

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
      state = state.add_input(expr_seq[i], expr_seq.mask)
      out_expr.append(state.output())

    out_expr = xnmt.ExpressionSequence(expr_list=out_expr, mask=expr_seq.mask)
    final_states = [models.states.FinalTransducerState(h, c) for h, c in zip(state.h(), state.c())]
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
    self.num_layers = layers
    self.hidden_dim = hidden_dim
    self.dropout_rate = dropout
    self.weightnoise_std = weightnoise_std
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
    bwd_encode = self.backward_layers.transduce(xnmt.ReversedExpressionSequence(es))
    fwd_es = fwd_encode.encode_seq
    bwd_es = bwd_encode.encode_seq
    fwd_final_states = fwd_encode.encoder_final_states
    bwd_final_states = bwd_encode.encoder_final_states

    expr_list = [dy.concatenate([fw, bw]) for fw, bw in zip(fwd_es, reversed(bwd_es))]
    final_states = [models.FinalTransducerState(main_expr=dy.concatenate([fwd_final_state.main_expr(),
                                                                          bwd_final_state.main_expr()]),
                                                cell_expr=dy.concatenate([fwd_final_state.cell_expr(),
                                                                          bwd_final_state.cell_expr()])) \
                    for fwd_final_state, bwd_final_state in zip(fwd_final_states, bwd_final_states)]


    expr_seq = xnmt.ExpressionSequence(expr_list=expr_list, mask=es.mask)
    return models.EncoderState(expr_seq, final_states)

