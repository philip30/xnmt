from typing import List

import dynet as dy
import xnmt
import xnmt.models as models

from typing import Optional, Dict

class RewardValue(object):

  def __init__(self, value: float, data: Optional[Dict[str, float]] = None):
    self.value = value
    self.data = data


class Policy(object):

  def sample_actions(self, states: xnmt.ExpressionSequence, argmax=False, sample_pp=None, predefined_action=None):
    actions = []
    for i, (state, predef_act) in enumerate(zip(states, predefined_action)):
      mask = states.mask[i] if states.mask[i] is not None else None
      actions.append(self.sample_action(state, argmax, sample_pp, predefined_action, mask))
    return actions

  def sample_action(self, state, argmax=False, sample_pp=None, predefined_actions=None, mask=None) -> models.states.SearchAction:
    """
    state: Input state.
    argmax: Whether to perform argmax or sampling.
    sample_pp: Stands for sample post_processing.
               Every time the sample are being drawn, this method will be invoked with sample_pp(sample).
    predefined_actions: Whether to forcefully the network to assign the action value to some predefined actions.
                        This predefined actions can be from the gold distribution or some probability priors.
                        It should be calculated from the outside.
    """
    policy = dy.log_softmax(self.input_state(state))

    # Select actions
    if predefined_actions is not None and len(predefined_actions) != 0:
      actions = predefined_actions
    else:
      if argmax:
        actions = policy.tensor_value().argmax().as_numpy()[0]
      else:
        actions = policy.tensor_value().categorical_sample_log_prob().as_numpy()[0]
      if len(actions.shape) == 0:
        actions = [actions]
    # Post Processing
    if sample_pp is not None:
      actions = sample_pp(actions)
    # Return
    return xnmt.models.states.SearchAction(actions, policy, state, mask)

  def input_state(self, state):
    raise NotImplementedError()


class RewardCalculator(object):

  def calculate_reward(self, model, src, trg, ref) -> List[RewardValue]:
    assert len(src) == len(trg) and len (src) == len(ref)
    rewards = []
    for i in range(src.batch_size()):
      rewards.append(self.calculate_single_reward(i, model, src[i], trg[i], ref[i]))
    return rewards

  def calculate_single_reward(self, index, model, src, trg, ref) -> RewardValue:
    raise NotImplementedError("Must be implemented by sub children")


class PolicyConditionedModel(object):

  def __init__(self, policy_network: Policy, policy_train_oracle: bool, policy_test_oracle: bool):
    self.policy_network = policy_network
    self.policy_train_oracle = policy_train_oracle
    self.policy_test_oracle = policy_test_oracle

  def calc_policy_nll(self, src:xnmt.Batch, trg:xnmt.Batch, parent_model):
    raise NotImplementedError("must be implemented by subclasses")

  def create_trajectories(self, *args, **kwargs):
    raise NotImplementedError("Must be implemented by subclasses")

  def create_trajectory(
      self, src:xnmt.Batch, trg:xnmt.Batch, current_state: models.states.DecoderState=None,
      from_oracle: bool=False, force_decoding: bool=True, parent_model: models.basic.GeneratorModel=None):
    raise NotImplementedError("Must be implemented by subclasses")


class Baseline(object):

  def calculate_baseline(self, input_states: List[dy.Expression]):
    pass
