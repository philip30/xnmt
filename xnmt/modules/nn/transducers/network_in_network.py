from typing import Optional

import xnmt.models.templates
from xnmt import param_initializers
from xnmt.modules.nn import norms, transforms as modelparts_transforms
from xnmt.internal.persistence import Serializable, serializable_init, bare, Ref

from .modular import ModularSeqTransducer
from .transform import TransformSeqTransducer


class NinSeqTransducer(ModularSeqTransducer, Serializable):
  yaml_tag = "!NinSeqTransducer"
  """
  Network-in-network transducer following Lin et al. (2013): Network in Network; https://arxiv.org/pdf/1312.4400.pdf

  Here, this is a shared linear transformation across time steps, followed by batch normalization and a non-linearity.

  Args:
    input_dim: dimension of inputs
    hidden_dim: dimension of outputs
    downsample_by: if > 1, feed adjacent time steps to the linear projections to downsample the sequence
    param_init: how to initialize the projection matrix
    projection: automatically set
    batch_norm: automatically set
    nonlinearity: automatically set
  """
  @serializable_init
  def __init__(self,
               input_dim: int = Ref("exp_global.default_layer_dim"),
               hidden_dim: int = Ref("exp_global.default_layer_dim"),
               downsample_by: int = 1,
               param_init: xnmt.models.templates.ParamInitializer = Ref("exp_global.param_init", default=bare(
                 param_initializers.GlorotInitializer)),
               projection: Optional[TransformSeqTransducer] = None,
               batch_norm: Optional[norms.BatchNorm] = None,
               nonlinearity: Optional[TransformSeqTransducer] = None) -> None:
    self.projection = self.add_serializable_component("projection", projection,
                                                      lambda: TransformSeqTransducer(
                                                        modelparts_transforms.Linear(input_dim=input_dim*downsample_by,
                                                                                     output_dim=hidden_dim,
                                                                                     bias=False,
                                                                                     param_init=param_init),
                                                        downsample_by=downsample_by))
    self.batch_norm = self.add_serializable_component("batch_norm", batch_norm,
                                                      lambda: norms.BatchNorm(hidden_dim=hidden_dim,
                                                                              num_dim=2))
    self.nonlinearity = self.add_serializable_component("nonlinearity", nonlinearity,
                                                        lambda: TransformSeqTransducer(
                                                          modelparts_transforms.Cwise("rectify")
                                                        ))
    super.__init__(self.projection, self.batch_norm, self.nonlinearity)

