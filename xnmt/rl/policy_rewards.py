import numpy as np
import scipy.stats as stats

from typing import Dict, Optional, List

import xnmt.persistence as persistence
import xnmt.thirdparty.dl4mt_simul_trans.reward as simult_reward
import xnmt.eval.metrics as metrics
import xnmt.simultaneous.simult_translators as sim_translator
import xnmt.models.base as model_base

class RewardValue(object):

  def __init__(self, value: float, data: Optional[Dict[str, float]] = None):
    self.value = value
    self.data = data


class RewardCalculator(object):

  def calculate_reward(self, model, src, trg, ref) -> List[RewardValue]:
    assert len(src) == len(trg) and len (src) == len(ref)
    rewards = []
    for i in range(len(src)):
      rewards.append(self.calculate_single_reward(i, model, src[i], trg[i], ref[i]))
    return rewards

  def calculate_single_reward(self, index, model, src, trg, ref) -> RewardValue:
    raise NotImplementedError("Must be implemented by sub children")


class SentenceEvalMeasureReward(RewardCalculator, persistence.Serializable):

  yaml_tag = "!SentenceEvalMeasureReward"

  @persistence.serializable_init
  def __init__(self, eval_metrics: metrics.SentenceLevelEvaluator, inverse_eval=True):
    super().__init__()
    self.eval_metrics = eval_metrics
    self.inverse_eval = inverse_eval

  def calculate_single_reward(self, index, model, src, trg, ref) -> RewardValue:
    value = self.eval_metrics.evaluate_one_sent(ref, trg)

    if self.inverse_eval:
      value *= -1

    return RewardValue(value.value())


class SimNMTReward(RewardCalculator, persistence.Serializable):

  yaml_tag = "!SimNMTReward"

  @persistence.serializable_init
  def __init__(self):
    super().__init__()

  def calculate_single_reward(self, index, model: sim_translator.SimultaneousTranslator, src, trg, ref):
    action = model.actions[index]
    reward, bleu, delay, instant_reward = simult_reward.return_reward(trg, ref, action, src.len_unpadded())
    return RewardValue(reward, {"bleu": bleu, "delay": delay, "instant_reward": instant_reward})


class CompositeReward(RewardCalculator):

  @persistence.serializable_init
  def __init__(self, reward_calculators:List[RewardCalculator], weights:List[float]=None):
    self.reward_calculators = reward_calculators

    if weights is not None:
      assert len(weights) == len(reward_calculators)
    else:
      weights = [1] * len(reward_calculators)

    self.weights = np.array(weights)

  def calculate_single_reward(self, index, model, src, trg, ref):
    values = []
    data = {}
    for reward_calculator in self.reward_calculators:
      reward_value = reward_calculator.calculate_single_reward(index, model, src, trg, ref)
      values.append(reward_value.value)
      data.update(reward_value.data)
    return RewardValue(float(np.sum(np.asarray(values) * self.weights)), data)


class PoissonSrcLengthReward(RewardCalculator, persistence.Serializable):
  """
  A prior that tries the poisson probability of having a specific number of segment
  Given the expected number of segments.

  First we need to calculate the average number of characters inside its word from some corpus = lambda
  Then we expect the number of segments should be = #characters_in_input / lambda
  """

  def __init__(self, lmbd):
    super().__init__()
    self.lmbd = lmbd

  def calculate_single_reward(self, index, model: model_base.PolicyConditionedModel, src, trg, ref):
    value = np.log(stats.poisson.pmf(model.actions[index], mu=src.len_unpadded()/ self.lmbd))
    return RewardValue(value, {"src_len": src.len_unpadded(), "lmbd": self.lmbd})



