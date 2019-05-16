import dynet as dy
import numpy as np
import collections

from typing import Sequence

import xnmt
import xnmt.models as models
import xnmt.rl.baseline as baselines

class PolicyGradient(xnmt.Serializable):
  yaml_tag = "!PolicyGradient"

  yaml_tag = '!PolicyGradient'

  @xnmt.serializable_init
  def __init__(self, baseline_network: models.Baseline = xnmt.bare(baselines.TransformBaseline)):
    self.baseline_network = baseline_network

  def calc_loss(self,  actions: Sequence[models.SearchAction], rewards: Sequence[Sequence[float]]):

    assert len(actions) == len(rewards), "Unequal actions, rewards value"
    ones = np.ones(len(rewards[0])) # Make ones equal to the batch size

    input_states = [action.decoder_state.context for action in actions]
    policy_lls = [action.log_likelihood for action in actions]
    masks = [action.mask for action in actions]
    flags = [(1-mask if mask is not None else ones) for mask in masks]

    baseline = self.baseline_network.calculate_baseline(input_states)
    rewards = [dy.inputTensor(r, batched=True) for r in rewards]
    disc_rewards = [r-dy.nobackprop(b) for r, b in zip(rewards, baseline)]
    gradient = [dy.cmult(policy_ll, disc_reward) for policy_ll, disc_reward in zip(policy_lls, disc_rewards)]

    batch_gradient = [dy.cmult(g, dy.inputTensor(f, batched=True)) for g, f in zip(gradient, flags)]
    baseline_loss = [dy.cmult(dy.squared_distance(r, b), f) for r, b, f in zip(rewards, gradient, flags)]

    units = [collections.Counter({i:n for i, n in enumerate(f)}) for f in flags]
    ret_units = collections.Counter()
    list(ret_units.update(unit) for unit in units)
    ret_units = list(ret_units.values())

    return xnmt.FactoredLossExpr({
      "reinforce": xnmt.LossExpr(-dy.esum(batch_gradient), ret_units),
      "baseline V(x)": xnmt.LossExpr(dy.esum(baseline_loss), ret_units)
    })



## Z Normalization
## R = R - mean(R) / std(R)
#rewards = dy.concatenate(rewards, d=0)
#r_dim = rewards.dim()
#if self.z_normalization:
#  rewards_shape = dy.reshape(rewards, (r_dim[0][0], r_dim[1]))
#  rewards_mean = dy.mean_elems(rewards_shape)
#  rewards_std = dy.std_elems(rewards_shape) + 1e-20
#  rewards = (rewards - rewards_mean.value()) / rewards_std.value()
#rewards = dy.nobackprop(rewards)

