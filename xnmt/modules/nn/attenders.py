import math
import numpy as np
import dynet as dy

import xnmt
import xnmt.models as models


class MlpAttender(models.Attender, xnmt.Serializable):
  yaml_tag = "!MlpAttender"
  """
  Implements the attention model of Bahdanau et. al (2014)

  Args:
    input_dim: input dimension
    state_dim: dimension of state inputs
    hidden_dim: hidden MLP dimension
    param_init: how to initialize weight matrices
    bias_init: how to initialize bias vectors
  """
  @xnmt.serializable_init
  def __init__(self,
               input_dim: int = xnmt.default_layer_dim,
               state_dim: int = xnmt.default_layer_dim,
               hidden_dim: int = xnmt.default_layer_dim,
               param_init: xnmt.models.templates.ParamInitializer = xnmt.default_param_init,
               bias_init: xnmt.models.templates.ParamInitializer = xnmt.default_bias_init):
    self.input_dim = input_dim
    self.state_dim = state_dim
    self.hidden_dim = hidden_dim
    param_collection = xnmt.param_manager(self)
    self.pW = param_collection.add_parameters((hidden_dim, input_dim), init=param_init.initializer((hidden_dim, input_dim)))
    self.pV = param_collection.add_parameters((hidden_dim, state_dim), init=param_init.initializer((hidden_dim, state_dim)))
    self.pb = param_collection.add_parameters((hidden_dim,), init=bias_init.initializer((hidden_dim,)))
    self.pU = param_collection.add_parameters((1, hidden_dim), init=param_init.initializer((1, hidden_dim)))

  def initial_state(self, sent: xnmt.ExpressionSequence) -> models.AttenderState:
    sent_input = sent.as_tensor()
    inp_context = dy.affine_transform([self.pb, self.pW, sent_input])

    if inp_context.dim()[1] == 1:
      inp_context = dy.concatenate([inp_context], d=1)

    return models.AttenderState(sent, inp_context)

  def calc_attention(self, decoder_context: dy.Expression, attender_state: models.AttenderState) -> dy.Expression:
    curr_sent_mask = attender_state.curr_sent.mask
    h = dy.tanh(dy.colwise_add(attender_state.sent_context, self.pV * decoder_context))
    scores = dy.transpose(self.pU * h)
    if curr_sent_mask is not None:
      scores = curr_sent_mask.add_to_tensor_expr(scores, multiplicator=-xnmt.globals.INF)
    return dy.softmax(scores)


class DotAttender(models.Attender, xnmt.Serializable):
  yaml_tag = "!DotAttender"
  """
  Implements dot product attention of https://arxiv.org/abs/1508.04025
  Also (optionally) perform scaling of https://arxiv.org/abs/1706.03762

  Args:
    scale: whether to perform scaling
  """
  @xnmt.serializable_init
  def __init__(self, scale: bool = True):
    self.curr_sent = None
    self.scale = scale

  def initial_state(self, sent: xnmt.ExpressionSequence) -> models.AttenderState:
    return models.AttenderState(sent, dy.transpose(sent.as_tensor()))

  def calc_attention(self, decoder_context: dy.Expression, attender_state: models.AttenderState) -> dy.Expression:
    curr_sent_mask = attender_state.curr_sent.mask
    scores = attender_state.sent_context * decoder_context
    if self.scale:
      scores /= math.sqrt(decoder_context.dim()[0][0])
    if curr_sent_mask is not None:
      scores = curr_sent_mask.add_to_tensor_expr(scores, multiplicator=-xnmt.globals.INF)
    return dy.softmax(scores)


class BilinearAttender(models.Attender, xnmt.Serializable):
  yaml_tag = "!BilinearAttender"
  """
  Implements a bilinear attention, equivalent to the 'general' linear
  attention of https://arxiv.org/abs/1508.04025

  Args:
    input_dim: input dimension; if None, use exp_global.default_layer_dim
    state_dim: dimension of state inputs; if None, use exp_global.default_layer_dim
    param_init: how to initialize weight matrices; if None, use ``exp_global.param_init``
  """
  @xnmt.serializable_init
  def __init__(self,
               input_dim: int = xnmt.default_layer_dim,
               state_dim: int = xnmt.default_layer_dim,
               param_init: xnmt.ParamInitializer = xnmt.default_param_init):
    self.input_dim = input_dim
    self.state_dim = state_dim
    param_collection = xnmt.param_manager(self)
    self.pWa = param_collection.add_parameters((input_dim, state_dim), init=param_init.initializer((input_dim, state_dim)))
    self.curr_sent = None

  def initial_state(self, sent: xnmt.ExpressionSequence) -> models.AttenderState:
    return models.AttenderState(sent, sent.as_tensor())

  def calc_attention(self, decoder_context: dy.Expression, attender_state: models.AttenderState) -> dy.Expression:
    curr_sent_mask = attender_state.curr_sent.mask
    h = dy.transpose(decoder_context) * self.pWa
    scores = h * attender_state.sent_context
    if curr_sent_mask is not None:
      scores = curr_sent_mask.add_to_tensor_expr(scores, multiplicator=-xnmt.globals.INF)
    return dy.softmax(scores)


class LatticeBiasedMlpAttender(MlpAttender, xnmt.Serializable):
  yaml_tag = "!LatticeBiasedMlpAttender"
  """
  Modified MLP attention, where lattices are assumed as input and the attention is biased toward confident nodes.

  Args:
    input_dim: input dimension
    state_dim: dimension of state inputs
    hidden_dim: hidden MLP dimension
    param_init: how to initialize weight matrices
    bias_init: how to initialize bias vectors
  """
  @xnmt.serializable_init
  def __init__(self,
               input_dim: int = xnmt.default_layer_dim,
               state_dim: int = xnmt.default_layer_dim,
               hidden_dim: int = xnmt.default_layer_dim,
               param_init: xnmt.models.templates.ParamInitializer = xnmt.default_param_init,
               bias_init: xnmt.ParamInitializer = xnmt.default_bias_init) -> None:
    super().__init__(input_dim=input_dim, state_dim=state_dim, hidden_dim=hidden_dim, param_init=param_init,
                     bias_init=bias_init)

  def initial_state(self, sent: xnmt.ExpressionSequence) -> models.AttenderState:
    src_batch = xnmt.globals.singleton_global.src_batch
    cur_sent_bias = np.full((src_batch.sent_len(), 1, src_batch.batch_size()), -1e10)
    for i in range(len(src_batch.batch_size)):
      for node_id in src_batch[i].graph.topo_sort():
        cur_sent_bias[node_id, 0, i] = src_batch[i].graph[node_id].marginal_log_prob

    return models.AttenderState(sent, (sent.as_tensor(), cur_sent_bias))

  def calc_attention(self, decoder_context: dy.Expression, attender_state: models.AttenderState) -> dy.Expression:
    sent_context, curr_sent_bias = attender_state.sent_context
    curr_sent_mask = attender_state.curr_sent.mask

    h = dy.tanh(dy.colwise_add(sent_context, self.pV * decoder_context))
    scores = dy.transpose(self.pU * h) + dy.inputTensor(curr_sent_bias, batched=True)
    if curr_sent_mask is not None:
      scores = curr_sent_mask.add_to_tensor_expr(scores, multiplicator=-xnmt.globals.INF)

    return dy.softmax(scores)

