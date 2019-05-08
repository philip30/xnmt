from enum import Enum

import dynet as dy
import numpy as np

from xnmt import events, losses, param_initializers
from xnmt.modelparts import transforms
from xnmt.persistence import Ref, bare, Serializable, serializable_init
from xnmt.reports import Reportable


class PolicyGradientLoss(Serializable, Reportable):
  """
  (Sequence) policy gradient class. It holds a policy network that will perform a linear regression to the output_dim decision labels.
  This class works by calling the sample_action() that will sample some actions from the current policy, given the input state.

  Every time sample_action is called, the actions and the softmax are being recorded and then the network can be trained by passing the reward
  to the calc_loss function.

  Depending on the passed flags, currently it supports the calculation of additional losses of:
  - baseline: Linear regression that predicts future reward from the current input state
  - conf_penalty: The confidence penalty loss. Good when policy network are too confident.

  Args:
    sample: The number of samples being drawn from the policy network.
    use_baseline: Whether to turn on baseline reward discounting or not.
    weight: The weight of the reinforce_loss.
    output_dim: The size of the predicitions.
  """
  yaml_tag = '!PolicyGradientLoss'
  @serializable_init
  @events.register_xnmt_handler
  def __init__(self,
               policy_model=None,
               baseline_network=None,
               z_normalization=True,
               eps_greedy_prob=1.0):
    self.policy_model = policy_model
    self.baseline_network = baseline_network
    self.z_normalization = z_normalization

  def calc_loss(self, policy_reward, results={}):
    """
    Calc policy networks loss.
    """
    assert len(policy_reward) == len(self.states), "There should be a reward for every action taken"
    batch_size = self.states[0].dim()[1]
    loss = {}

    # Calculate the baseline loss of the reinforce loss for each timestep:
    # b = W_b * s + b_b
    # R = r - b
    # Also calculate the baseline loss
    # b = r_p (predicted)
    # loss_b = squared_distance(r_p - r_r)
    rewards = []
    baseline_loss = []
    units = np.zeros(batch_size)
    for i, state in enumerate(self.states):
      r_p = self.baseline.transform(dy.nobackprop(state))
      rewards.append(policy_reward[i] - r_p)
      if self.valid_pos[i] is not None:
        r_p = dy.pick_batch_elems(r_p, self.valid_pos[i])
        r_r = dy.pick_batch_elems(policy_reward[i], self.valid_pos[i])
        units[self.valid_pos[i]] += 1
      else:
        r_r = policy_reward[i]
        units += 1
      baseline_loss.append(dy.sum_batches(dy.squared_distance(r_p, r_r)))
    loss["rl_baseline"] = losses.LossExpr(dy.esum(baseline_loss), units)

    # Z Normalization
    # R = R - mean(R) / std(R)
    rewards = dy.concatenate(rewards, d=0)
    r_dim = rewards.dim()
    if self.z_normalization:
      rewards_shape = dy.reshape(rewards, (r_dim[0][0], r_dim[1]))
      rewards_mean = dy.mean_elems(rewards_shape)
      rewards_std = dy.std_elems(rewards_shape) + 1e-20
      rewards = (rewards - rewards_mean.value()) / rewards_std.value()
    rewards = dy.nobackprop(rewards)
    # Calculate Confidence Penalty
    if self.confidence_penalty:
      loss["rl_confpen"] = self.confidence_penalty.calc_loss(self.policy_lls)

    # Calculate Reinforce Loss
    # L = - sum([R-b] * pi_ll)
    reinf_loss = []
    units = np.zeros(batch_size)
    for i, (policy, action) in enumerate(zip(self.policy_lls, self.actions)):
      reward = dy.pick(rewards, i)
      ll = dy.pick_batch(policy, action)
      if self.valid_pos[i] is not None:
        ll = dy.pick_batch_elems(ll, self.valid_pos[i])
        reward = dy.pick_batch_elems(reward, self.valid_pos[i])
        units[self.valid_pos[i]] += 1
      else:
        units += 1
      reinf_loss.append(dy.sum_batches(dy.cmult(ll, reward)))
    loss["rl_reinf"] = losses.LossExpr(-dy.esum(reinf_loss), units)

    # Pack up + return
    return losses.FactoredLossExpr(loss)

