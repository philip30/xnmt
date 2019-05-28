import dynet as dy
import xnmt
import xnmt.models as models

from typing import Optional, Dict, Tuple, List

class RewardValue(object):

  def __init__(self, value: float, data: Optional[Dict[str, float]] = None):
    self.value = value
    self.data = data


class RewardCalculator(object):

  def calculate_reward(self, model, src, trg, ref) -> List[RewardValue]:
    assert len(src) == len(trg) and len (src) == len(ref)
    rewards = []
    for i in range(src.batch_size()):
      rewards.append(self.calculate_single_reward(i, model, src[i], trg[i], ref[i]))
    return rewards

  def calculate_single_reward(self, index, model, src, trg, ref) -> RewardValue:
    raise NotImplementedError("Must be implemented by sub children")


class PolicyAgentState(models.UniDirectionalState):

  def __init__(self, src: xnmt.Batch, policy_state: models.UniDirectionalState):
    self.src = src
    self.policy_state = policy_state

  def add_input(self, word: dy.Expression, mask: Optional[xnmt.Mask] = None) -> 'PolicyAgentState':
    return PolicyAgentState(self.src, self.policy_state.add_input(word, mask))

  def output(self):
    return self.policy_state.output()


class DoubleAttentionPolicyAgentState(PolicyAgentState):
  def __init__(self,
               src: xnmt.Batch,
               policy_state: models.UniDirectionalState,
               encoder_state: models.AttenderState,
               decoder_state: models.AttenderState):
    super().__init__(src, policy_state)
    self.encoder_state = encoder_state
    self.decoder_state = decoder_state

  def add_input(self, word: dy.Expression, mask: Optional[xnmt.Mask] = None) -> 'DoubleAttentionPolicyAgentState':
    return DoubleAttentionPolicyAgentState(self.src, self.policy_state.add_input(word, mask),
                                           self.encoder_state, self.decoder_state)


class PolicyAgent(object):
  __ACTIONS__= []

  def initial_state(self, src: xnmt.Batch) -> PolicyAgentState:
    raise NotImplementedError()

  def finish_generating(self, states: models.UniDirectionalState) -> bool:
    raise NotImplementedError()

  def next_action(self, state: models.UniDirectionalState) \
      -> Tuple[models.SearchAction, PolicyAgentState]:
    raise NotImplementedError()

  @classmethod
  def total_actions(cls):
    return len(cls.__ACTIONS__)
