import numpy as np
import scipy.stats as stats

from typing import List

import xnmt.thirdparty.dl4mt_simul_trans.reward as simult_reward

import xnmt
import xnmt.eval.evaluators as evaluators
import xnmt.models as models


class SentenceEvalMeasureReward(models.RewardCalculator,xnmt.Serializable):
  yaml_tag = "!SentenceEvalMeasureReward"
  @xnmt.serializable_init
  def __init__(self, eval_metrics: evaluators.SentenceLevelEvaluator, inverse_eval=True):
    super().__init__()
    self.eval_metrics = eval_metrics
    self.inverse_eval = inverse_eval

  def calculate_single_reward(self, index, model, src, trg, ref) -> models.RewardValue:
    value = self.eval_metrics.evaluate_one_sent(ref, trg)

    if self.inverse_eval:
      value *= -1

    return models.RewardValue(value.value())


class SimNMTReward(models.RewardCalculator, xnmt.Serializable):
  yaml_tag = "!SimNMTReward"
  @xnmt.serializable_init
  def __init__(self):
    super().__init__()

  def calculate_single_reward(self, index, model, src, trg, ref):
    action = model.actions[index]
    reward, bleu, delay, instant_reward = simult_reward.return_reward(trg, ref, action, src.len_unpadded())
    return models.RewardValue(reward, {"bleu": bleu, "delay": delay, "instant_reward": instant_reward})


class CompositeReward(models.RewardCalculator):

  @xnmt.serializable_init
  def __init__(self, reward_calculators:List[models.RewardCalculator], weights:List[float]=None):
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
    return models.RewardValue(float(np.sum(np.asarray(values) * self.weights)), data)


class PoissonSrcLengthReward(models.RewardCalculator, xnmt.Serializable):
  yaml_tag = "!PoissonSrcLengthReward"
  """
  A prior that tries the poisson probability of having a specific number of segment
  Given the expected number of segments.

  First we need to calculate the average number of characters inside its word from some corpus = lambda
  Then we expect the number of segments should be = #characters_in_input / lambda
  """
  @xnmt.serializable_init
  def __init__(self, lmbd):
    super().__init__()
    self.lmbd = lmbd

  def calculate_single_reward(self, index, model: models.PolicyConditionedModel, src, trg, ref):
    value = np.log(stats.poisson.pmf(model.actions[index], mu=src.len_unpadded()/ self.lmbd))
    return models.RewardValue(value, {"src_len": src.len_unpadded(), "lmbd": self.lmbd})



