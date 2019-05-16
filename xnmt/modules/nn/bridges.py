import dynet as dy
from typing import List, Optional, Sequence

import xnmt
import xnmt.models as models
import xnmt.modules.nn.transforms as transforms


class NoBridge(models.Bridge, xnmt.Serializable):
  yaml_tag = "!NoBridge"
  """
  This bridge initializes the decoder with zero vectors, disregarding the encoder final states.

  Args:
    dec_layers: number of decoder layers to initialize
    dec_dim: hidden dimension of decoder states
  """
  @xnmt.serializable_init
  def __init__(self,
               dec_layers: int = 1,
               dec_dim: int = xnmt.default_layer_dim):
    self.dec_layers = dec_layers
    self.dec_dim = dec_dim

  def decoder_init(self, enc_final_states: Sequence[models.FinalTransducerState]) -> List[dy.Expression]:
    return None


class CopyBridge(models.Bridge, xnmt.Serializable):
  yaml_tag = "!CopyBridge"
  """
  This bridge copies final states from the encoder to the decoder initial states.
  Requires that:
  - encoder / decoder dimensions match for every layer
  - num encoder layers >= num decoder layers (if unequal, we disregard final states at the encoder bottom)

  Args:
    dec_layers: number of decoder layers to initialize
    dec_dim: hidden dimension of decoder states
  """
  @xnmt.serializable_init
  def __init__(self,
               dec_layers: int = 1,
               dec_dim: int = xnmt.default_layer_dim):
    self.dec_layers = dec_layers
    self.dec_dim = dec_dim

  def decoder_init(self, enc_final_states: Sequence[models.FinalTransducerState]) -> List[dy.Expression]:
    if self.dec_layers > len(enc_final_states):
      raise RuntimeError("CopyBridge requires dec_layers <= len(enc_final_states), but got {} and {}".format(
                         self.dec_layers, len(enc_final_states)))
    if enc_final_states[0].main_expr().dim()[0][0] != self.dec_dim:
      raise RuntimeError("CopyBridge requires enc_dim == dec_dim, but got {} and {}".format(
                         enc_final_states[0].main_expr().dim()[0][0], self.dec_dim))

    cell_exprs = [enc_state.cell_expr() for enc_state in enc_final_states[-self.dec_layers:]]
    main_exprs = [enc_state.main_expr() for enc_state in enc_final_states[-self.dec_layers:]]

    return cell_exprs + main_exprs


class LinearBridge(models.Bridge, xnmt.Serializable):
  yaml_tag = "!LinearBridge"
  """
  This bridge does a linear transform of final states from the encoder to the decoder initial states.
  Requires that  num encoder layers >= num decoder layers (if unequal, we disregard final states at the encoder bottom)

  Args:
    dec_layers: number of decoder layers to initialize
    enc_dim: hidden dimension of encoder states
    dec_dim: hidden dimension of decoder states
    param_init: how to initialize weight matrices; if None, use ``exp_global.param_init``
    bias_init: how to initialize bias vectors; if None, use ``exp_global.bias_init``
    projector: linear projection (created automatically)
  """
  @xnmt.serializable_init
  def __init__(self,
               dec_layers: int = 1,
               enc_dim: int = xnmt.default_layer_dim,
               dec_dim: int = xnmt.default_layer_dim,
               param_init: xnmt.models.templates.ParamInitializer = xnmt.default_param_init,
               bias_init: xnmt.models.templates.ParamInitializer = xnmt.default_bias_init,
               projector: Optional[transforms.Linear]=None):
    self.dec_layers = dec_layers
    self.enc_dim = enc_dim
    self.dec_dim = dec_dim
    self.projector = self.add_serializable_component("projector", projector,
                                                     lambda: transforms.Linear(input_dim=self.enc_dim,
                                                                               output_dim=self.dec_dim,
                                                                               param_init=param_init,
                                                                               bias_init=bias_init))

  def decoder_init(self, enc_final_states: Sequence[models.FinalTransducerState]) -> List[dy.Expression]:
    if self.dec_layers > len(enc_final_states):
      raise RuntimeError(
        f"LinearBridge requires dec_layers <= len(enc_final_states), but got {self.dec_layers} and {len(enc_final_states)}")
    if enc_final_states[0].main_expr().dim()[0][0] != self.enc_dim:
      raise RuntimeError(
        f"LinearBridge requires enc_dim == {self.enc_dim}, but got {enc_final_states[0].main_expr().dim()[0][0]}")

    cell_exprs = [self.projector.transform(enc_state.main_expr()) for enc_state in enc_final_states[-self.dec_layers:]]
    main_exprs = [dy.tanh(cell_expr) for cell_expr in cell_exprs]

    return cell_exprs + main_exprs
