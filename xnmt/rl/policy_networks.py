import dynet as dy
import xnmt
import xnmt.models as models
import xnmt.modules.nn as nn

from typing import Any, List, Optional


class PolicyNetwork(models.Decoder):

  def __init__(self,
               transform: models.Transform = xnmt.bare(nn.NonLinear),
               scorer: models.Scorer = xnmt.bare(nn.Softmax)):
    self.transform = transform
    self.scorer = scorer

  def initial_state(self, src: xnmt.Batch) -> models.UniDirectionalState:
    raise NotImplementedError()

  def add_input(self, input_expr: dy.Expression, prev_state:models.UniDirectionalState) -> models.UniDirectionalState:
    raise NotImplementedError()

  def calc_loss(
      self, dec_state: models.PolicyAgentState, ref_action: xnmt.Batch):
    return self.scorer.calc_loss(self.transform.transform(dec_state.output()), ref_action)

  def best_k(self, dec_state: models.UniDirectionalState, k: int, normalize_scores=False) -> List[models.SearchAction]:
    best_k = self.scorer.best_k(self.transform.transform(dec_state.output()), k, normalize_scores)
    ret  = [models.SearchAction(dec_state, best_word, dy.pick_batch(log_softmax, best_word), log_softmax, None) \
            for best_word, log_softmax in best_k]
    return ret

  def pick_oracle(self, oracle, dec_state: models.UniDirectionalState):
    log_prob = self.scorer.calc_log_probs(self.transform.transform(dec_state.output()))
    return [models.SearchAction(dec_state, oracle, dy.pick_batch(log_prob, oracle), log_prob, None)]


  def sample(self, dec_state: models.UniDirectionalState, n: int, temperature=1.0):
    sample_k = self.scorer.sample(self.transform.transform(dec_state.output()), n, temperature)
    ret  = [models.SearchAction(dec_state, best_word, dy.pick_batch(log_softmax, best_word), log_softmax, None) \
            for best_word, log_softmax in sample_k]
    return ret

  def finish_generating(self, dec_output: Any, dec_state: models.UniDirectionalState):
    raise ValueError("Should not call finish generating from this model")


class TransformPolicyNetwork(PolicyNetwork, xnmt.Serializable):
  yaml_tag = "!TransformPolicyNetwork"

  @xnmt.serializable_init
  def __init__(self,
               transform: models.Transform = xnmt.bare(nn.NonLinear),
               scorer: models.Scorer = xnmt.bare(nn.Softmax)):
    super().__init__(transform=transform, scorer=scorer)

  def initial_state(self, src: xnmt.Batch):
    return models.IdentityUniDirectionalState()

  def add_input(self, input_expr, previous_state):
    return models.IdentityUniDirectionalState(input_expr)


class RecurrentPolicyNetwork(PolicyNetwork, xnmt.Serializable):
  yaml_tag = "!RecurrentPolicyNetwork"

  @xnmt.serializable_init
  def __init__(self,
               transform: models.Transform = xnmt.bare(nn.NonLinear),
               scorer: models.Scorer = xnmt.bare(nn.Softmax),
               rnn: models.UniDiSeqTransducer = xnmt.bare(nn.UniLSTMSeqTransducer)):
    super().__init__(transform=transform, scorer=scorer)
    self.rnn = rnn

  def initial_state(self, src: xnmt.Batch):
    return self.rnn.initial_state()

  def add_input(self, input_expr: dy.Expression, previous_state: models.UniDirectionalState):
    return previous_state.add_input(input_expr, None)
