from typing import Optional
import numbers

import dynet as dy
import numpy as np
import xnmt
import xnmt.models as models

from xnmt.internal.param_collections import ParamManager


"""
The purpose of this module is mostly to expose the DyNet trainers to YAML serialization,
but may also be extended to customize optimizers / training schedules
"""
class SGDTrainer(models.XnmtOptimizer, xnmt.Serializable):
  yaml_tag = "!SGDTrainer"
  """
  Stochastic gradient descent trainer

  This trainer performs stochastic gradient descent, the goto optimization procedure for neural networks.

  Args:
    e0: Initial learning rate
    skip_noisy: keep track of a moving average and a moving standard deviation of the log of the gradient norm
                          values, and abort a step if the norm of the gradient exceeds four standard deviations of the
                          moving average. Reference: https://arxiv.org/pdf/1804.09849.pdf
  """
  @xnmt.serializable_init
  def __init__(self, e0: numbers.Real = 0.1, skip_noisy: bool = False) -> None:
    super().__init__(optimizer=dy.SimpleSGDTrainer(ParamManager.global_collection(), e0),
                     skip_noisy=skip_noisy)


class MomentumSGDTrainer(models.XnmtOptimizer, xnmt.Serializable):
  yaml_tag = "!MomentumSGDTrainer"
  """
  Stochastic gradient descent with momentum

  This is a modified version of the SGD algorithm with momentum to stablize the gradient trajectory.

  Args:
    e0: Initial learning rate
    mom: Momentum
    skip_noisy: keep track of a moving average and a moving standard deviation of the log of the gradient norm
                          values, and abort a step if the norm of the gradient exceeds four standard deviations of the
                          moving average. Reference: https://arxiv.org/pdf/1804.09849.pdf
  """
  @xnmt.serializable_init
  def __init__(self, e0: numbers.Real = 0.01, mom: numbers.Real = 0.9, skip_noisy: bool = False) -> None:
    super().__init__(optimizer=dy.MomentumSGDTrainer(ParamManager.global_collection(), e0, mom),
                     skip_noisy=skip_noisy)


class AdagradTrainer(models.XnmtOptimizer, xnmt.Serializable):
  yaml_tag = "!AdagradTrainer"
  """
  Adagrad optimizer

  The adagrad algorithm assigns a different learning rate to each parameter.

  Args:
    e0: Initial learning rate
    eps: Epsilon parameter to prevent numerical instability
    skip_noisy: keep track of a moving average and a moving standard deviation of the log of the gradient norm
                          values, and abort a step if the norm of the gradient exceeds four standard deviations of the
                          moving average. Reference: https://arxiv.org/pdf/1804.09849.pdf
  """
  @xnmt.serializable_init
  def __init__(self, e0: numbers.Real = 0.1, eps: numbers.Real = 1e-20, skip_noisy: bool = False) -> None:
    super().__init__(optimizer=dy.AdagradTrainer(ParamManager.global_collection(), e0, eps=eps),
                     skip_noisy=skip_noisy)


class AdadeltaTrainer(models.XnmtOptimizer, xnmt.Serializable):
  yaml_tag = "!AdadeltaTrainer"
  """
  AdaDelta optimizer

  The AdaDelta optimizer is a variant of Adagrad aiming to prevent vanishing learning rates.

  Args:
    eps: Epsilon parameter to prevent numerical instability
    rho: Update parameter for the moving average of updates in the numerator
    skip_noisy: keep track of a moving average and a moving standard deviation of the log of the gradient norm
                          values, and abort a step if the norm of the gradient exceeds four standard deviations of the
                          moving average. Reference: https://arxiv.org/pdf/1804.09849.pdf
  """
  @xnmt.serializable_init
  def __init__(self, eps: numbers.Real = 1e-6, rho: numbers.Real = 0.95, skip_noisy: bool = False) -> None:
    super().__init__(optimizer=dy.AdadeltaTrainer(ParamManager.global_collection(), eps, rho),
                     skip_noisy=skip_noisy)


class AdamTrainer(models.XnmtOptimizer, xnmt.Serializable):
  yaml_tag = "!AdamTrainer"
  """
  Adam optimizer

  The Adam optimizer is similar to RMSProp but uses unbiased estimates of the first and second moments of the gradient

  Args:
    alpha: Initial learning rate
    beta_1: Moving average parameter for the mean
    beta_2: Moving average parameter for the variance
    eps: Epsilon parameter to prevent numerical instability
    skip_noisy: keep track of a moving average and a moving standard deviation of the log of the gradient norm
                          values, and abort a step if the norm of the gradient exceeds four standard deviations of the
                          moving average. Reference: https://arxiv.org/pdf/1804.09849.pdf
  """
  @xnmt.serializable_init
  def __init__(self,
               alpha: numbers.Real = 0.001,
               beta_1: numbers.Real = 0.9,
               beta_2: numbers.Real = 0.999,
               eps: numbers.Real = 1e-8,
               skip_noisy: bool = False) -> None:
    super().__init__(optimizer=dy.AdamTrainer(ParamManager.global_collection(), alpha, beta_1, beta_2, eps),
                     skip_noisy=skip_noisy)


class NoamTrainer(models.XnmtOptimizer, xnmt.Serializable):
  yaml_tag = "!NoamTrainer"
  """
  Proposed in the paper "Attention is all you need" (https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf) [Page 7, Eq. 3]
  In this the learning rate of Adam Optimizer is increased for the first warmup steps followed by a gradual decay

  Args:
    alpha:
    dim:
    warmup_steps:
    beta_1:
    beta_2:
    eps:
    skip_noisy: keep track of a moving average and a moving standard deviation of the log of the gradient norm
                          values, and abort a step if the norm of the gradient exceeds four standard deviations of the
                          moving average. Reference: https://arxiv.org/pdf/1804.09849.pdf
  """
  @xnmt.serializable_init
  def __init__(self,
               alpha: numbers.Real = 1.0,
               dim: numbers.Integral = 512,
               warmup_steps: Optional[numbers.Integral] = 4000,
               beta_1: numbers.Real = 0.9,
               beta_2: numbers.Real = 0.98,
               eps: numbers.Real = 1e-9,
               skip_noisy: bool = False) -> None:
    super().__init__(optimizer=dy.AdamTrainer(ParamManager.global_collection(),
                                    alpha=alpha,
                                    beta_1=beta_1,
                                    beta_2=beta_2,
                                    eps=eps),
                     skip_noisy=skip_noisy)
    self.dim = dim
    self.warmup_steps = warmup_steps
    self.steps = 0

  def update(self) -> None:
    self.steps += 1
    if self.warmup_steps:
      decay = (self.dim ** (-0.5)) * np.min([self.steps ** (-0.5), self.steps * (self.warmup_steps ** (-1.5))])
    else:
      decay = (self.dim ** (-0.5)) * self.steps ** (-0.5)
    self.optimizer.learning_rate = 1. * decay
    super().update()

    if self.steps % 200 == 0:
      xnmt.logger.info('> Optimizer Logging')
      xnmt.logger.info('  Steps=%d, learning_rate=%.2e' % (self.steps, self.optimizer.learning_rate))


class DummyTrainer(models.XnmtOptimizer, xnmt.Serializable):
  yaml_tag = "!DummyTrainer"
  """
  A dummy trainer that does not perform any parameter updates.
  """
  @xnmt.serializable_init
  def __init__(self) -> None:
    pass

  def update(self) -> None:
    pass

  def status(self) -> None:
    pass

  def set_clip_threshold(self, thr) -> None:
    pass

  def get_clip_threshold(self) -> None:
    pass

  def restart(self) -> None:
    pass

  @property
  def learning_rate(self):
    return 1.0
  @learning_rate.setter
  def learning_rate(self, value):
    pass

