import dynet as dy
from typing import Optional, Sequence

import xnmt
import xnmt.models as models
import xnmt.modules as modules


class IdentityTransform(models.Transform, xnmt.Serializable):
  """
  Identity transform. For use when you think it might be a better idea to
  not perform a specific transform in a place where you would normally do one.
  """
  yaml_tag = "!IdentityTransform"

  def transform(self, input_expr: dy.Expression) -> dy.Expression:
    return input_expr


class Linear(models.Transform, xnmt.Serializable):
  """
  Linear projection with optional bias.

  Args:
    input_dim: input dimension
    output_dim: hidden dimension
    bias: whether to add a bias
    param_init: how to initialize weight matrices
    bias_init: how to initialize bias vectors
  """
  yaml_tag = "!Linear"

  @xnmt.serializable_init
  def __init__(self,
               input_dim: int = xnmt.default_layer_dim,
               output_dim: int = xnmt.default_layer_dim,
               bias: bool = True,
               param_init: xnmt.models.templates.ParamInitializer = xnmt.default_param_init,
               bias_init: xnmt.models.templates.ParamInitializer = xnmt.default_bias_init):
    self.bias = bias
    self.input_dim = input_dim
    self.output_dim = output_dim

    model = xnmt.param_manager(self)
    self.W1 = model.add_parameters((output_dim, input_dim), init=param_init.initializer((output_dim, input_dim)))
    if self.bias:
      self.b1 = model.add_parameters((output_dim,), init=bias_init.initializer((output_dim,)))

  def transform(self, input_expr: dy.Expression) -> dy.Expression:
    if self.bias:
      return dy.affine_transform([self.b1, self.W1, input_expr])
    else:
      return self.W1 * input_expr


class NonLinear(models.Transform, xnmt.Serializable):
  """
  Linear projection with optional bias and non-linearity.

  Args:
    input_dim: input dimension
    output_dim: hidden dimension
    bias: whether to add a bias
    activation: One of ``tanh``, ``relu``, ``sigmoid``, ``elu``, ``selu``, ``asinh`` or ``identity``.
    param_init: how to initialize weight matrices
    bias_init: how to initialize bias vectors
  """

  yaml_tag = "!NonLinear"

  @xnmt.serializable_init
  def __init__(self,
               input_dim: int = xnmt.default_layer_dim,
               output_dim: int = xnmt.default_layer_dim,
               bias: bool = True,
               activation: str = 'tanh',
               param_init: xnmt.models.templates.ParamInitializer = xnmt.default_param_init,
               bias_init: xnmt.models.templates.ParamInitializer = xnmt.default_bias_init):
    self.bias = bias
    self.output_dim = output_dim
    self.input_dim = input_dim
    self.activation = modules.activations.dynet_activation_from_string(activation)

    model = xnmt.param_manager(self)
    self.W1 = model.add_parameters((self.output_dim, self.input_dim), init=param_init.initializer((self.output_dim, self.input_dim)))
    if self.bias:
      self.b1 = model.add_parameters((self.output_dim,), init=bias_init.initializer((self.output_dim,)))

  def transform(self, input_expr: dy.Expression) -> dy.Expression:
    if self.bias:
      return self.activation(dy.affine_transform([self.b1, self.W1, input_expr]))
    else:
      return self.activation(self.W1 * input_expr)


# TODO: can we come up with a more elegant way to handle things that doesn't require this?
#       currently this is necessary because of this: https://github.com/neulab/xnmt/issues/441#issuecomment-400051066
class AuxNonLinear(NonLinear, xnmt.Serializable):
  """
  NonLinear with an additional auxiliary input.

  Args:
    input_dim: input dimension
    output_dim: hidden dimension
    aux_input_dim: auxiliary input dimension.
                   The actual input dimension is aux_input_dim + input_dim.
                   This is useful for when you want to do something like input feeding.
    bias: whether to add a bias
    activation: One of ``tanh``, ``relu``, ``sigmoid``, ``elu``, ``selu``, ``asinh`` or ``identity``.
    param_init: how to initialize weight matrices
    bias_init: how to initialize bias vectors
  """

  yaml_tag = "!AuxNonLinear"

  @xnmt.serializable_init
  def __init__(self,
               input_dim: int = xnmt.default_layer_dim,
               output_dim: int = xnmt.default_layer_dim,
               aux_input_dim: int = xnmt.default_layer_dim,
               bias: bool = True,
               activation: str = 'tanh',
               param_init: xnmt.models.templates.ParamInitializer = xnmt.default_param_init,
               bias_init: xnmt.models.templates.ParamInitializer = xnmt.default_bias_init):
    original_input_dim = input_dim
    input_dim += aux_input_dim
    super().__init__(
      input_dim=input_dim,
      output_dim=output_dim,
      bias=bias,
      activation=activation,
      param_init=param_init,
      bias_init=bias_init
    )
    self.save_processed_arg("input_dim", original_input_dim)


class MLP(models.Transform, xnmt.Serializable):
  """
  A multi-layer perceptron. Defined as one or more NonLinear transforms of equal hidden
  dimension and type, then a Linear transform to the output dimension.
  """
  yaml_tag = "!MLP"

  @xnmt.serializable_init
  def __init__(self,
               input_dim: int = xnmt.default_layer_dim,
               hidden_dim: int = xnmt.default_layer_dim,
               output_dim: int = xnmt.default_layer_dim,
               bias: bool = True,
               activation: str = 'tanh',
               hidden_layers: int = 1,
               param_init: xnmt.models.templates.ParamInitializer = xnmt.default_param_init,
               bias_init: xnmt.models.templates.ParamInitializer = xnmt.default_bias_init,
               layers: Optional[Sequence[models.Transform]] = None):
    self.layers = self.add_serializable_component("layers",
                                                  layers,
                                                  lambda: MLP._create_layers(num_layers=hidden_layers,
                                                                             input_dim=input_dim,
                                                                             hidden_dim=hidden_dim,
                                                                             output_dim=output_dim,
                                                                             bias=bias,
                                                                             activation=activation,
                                                                             param_init=param_init,
                                                                             bias_init=bias_init))

  @staticmethod
  def _create_layers(num_layers: int, input_dim: int, hidden_dim: int,
                     output_dim: int, bias: bool, activation: str,
                     param_init: xnmt.ParamInitializer,
                     bias_init: xnmt.ParamInitializer) -> Sequence[models.Transform]:
    layers = []
    if num_layers > 0:
      layers = [NonLinear(input_dim=input_dim, output_dim=hidden_dim, bias=bias, activation=activation,
                          param_init=param_init, bias_init=bias_init)]
      layers += [NonLinear(input_dim=hidden_dim, output_dim=hidden_dim, bias=bias, activation=activation,
                           param_init=param_init, bias_init=bias_init) for _ in range(1, num_layers)]
    layers += [Linear(input_dim=hidden_dim if num_layers > 0 else input_dim,
                      output_dim=output_dim,
                      bias=bias,
                      param_init=param_init,
                      bias_init=bias_init)]
    return layers

  def transform(self, expr: dy.Expression) -> dy.Expression:
    for layer in self.layers:
      expr = layer.transform(expr)
    return expr


class Cwise(models.Transform, xnmt.Serializable):
  """
  A component-wise transformation that can be an arbitrary unary DyNet operation.

  Args:
    op: arbitrary unary DyNet node
  """
  yaml_tag = "!Cwise"
  @xnmt.serializable_init
  def __init__(self, op: str = "rectify") -> None:
    self.op = getattr(dy, op, None)
    if not self.op:
      raise ValueError(f"DyNet does not have an operation '{op}'.")

  def transform(self, input_expr: dy.Expression) -> dy.Expression:
    return self.op(input_expr)
