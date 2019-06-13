import math
import numpy as np
import dynet as dy

import xnmt
import xnmt.models as models
import xnmt.modules.nn as nn

from typing import Optional


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

  def not_empty_initial_state(self, sent: xnmt.ExpressionSequence, value: xnmt.ExpressionSequence) -> models.AttenderState:
    sent_input = sent.as_tensor()
    if value is not None: value = value.as_tensor()
    inp_context = dy.affine_transform([self.pb, self.pW, sent_input])

    if inp_context.dim()[1] == 1:
      inp_context = dy.concatenate([inp_context], d=1)

    return models.AttenderState(sent_input, inp_context, sent.mask, curr_value=value)

  def calc_scores(self, decoder_context: dy.Expression, attender_state: models.AttenderState) -> dy.Expression:
    sent_context = attender_state.sent_context
    h = dy.tanh(dy.colwise_add(sent_context, self.pV * decoder_context))
    return dy.transpose(self.pU * h)


class DotAttender(models.Attender, xnmt.Serializable):
  yaml_tag = "!DotAttender"
  """
  Implements dot product attention of https://arxiv.org/abs/1508.04025
  Also (optionally) perform scaling of https://arxiv.org/abs/1706.03762

  Args:
    scale: whether to perform scaling
  """
  @xnmt.serializable_init
  def __init__(self, scale: bool = True, layer_dim=xnmt.default_layer_dim):
    self.curr_sent = None
    self.is_scale = scale
    self.scale = math.sqrt(layer_dim)

  def not_empty_initial_state(self, sent: xnmt.ExpressionSequence, value: xnmt.ExpressionSequence) -> models.AttenderState:
    sent_input = sent.as_tensor()
    if value is not None: value = value.as_tensor()
    return models.AttenderState(sent_input, sent_input, sent.mask, curr_value=value)

  def calc_scores(self, decoder_context: dy.Expression, attender_state: models.AttenderState) -> dy.Expression:
    scores = dy.transpose(attender_state.curr_sent) * decoder_context
    if self.is_scale:
      scores = scores * (1/self.scale)
    return scores


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

  def not_empty_initial_state(self, sent: xnmt.ExpressionSequence, value: xnmt.ExpressionSequence) -> models.AttenderState:
    if value is not None: value = value.as_tensor()
    return models.AttenderState(sent.as_tensor(), sent.as_tensor(), sent.mask, value)

  def calc_scores(self, decoder_context: dy.Expression, attender_state: models.AttenderState) -> dy.Expression:
    return dy.transpose(decoder_context) * self.pWa * attender_state.curr_sent
