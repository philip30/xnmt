import itertools
import xnmt
import numpy as np
import dynet as dy
import xnmt.models.states as states

from typing import Sequence, Optional, List


class TrainableModel(object):
  """
  A template class for a basic trainable model, implementing a loss function.
  """

  def calc_nll(self, *args, **kwargs) -> dy.Expression:
    """Calculate loss based on input-output pairs.
    Losses are accumulated only across unmasked timesteps in each batch element.
    Arguments are to be defined by subclasses
    Returns:
      A (possibly batched) expression representing the loss.
    """

class UnconditionedModel(TrainableModel):
  """
  A template class for trainable model that computes target losses without conditioning on other inputs.
  """

  def calc_nll(self, trg: xnmt.Batch) -> xnmt.LossExpr:
    """Calculate loss based on target inputs.

    Losses are accumulated only across unmasked timesteps in each batch element.

    Args:
      trg: The target, a sentence or a batch of sentences.

    Returns:
      A (possibly batched) expression representing the loss.
    """

class ConditionedModel(TrainableModel):

  """
  A template class for a trainable model that computes target losses conditioned on a source input.
  """

  def calc_nll(self, src: xnmt.Batch, trg: xnmt.Batch) -> xnmt.LossExpr:
    """Calculate loss based on input-output pairs.

    Losses are accumulated only across unmasked timesteps in each batch element.

    Args:
      src: The source, a sentence or a batch of sentences.
      trg: The target, a sentence or a batch of sentences.

    Returns:
      A (possibly batched) expression representing the loss.
    """
    raise NotImplementedError("must be implemented by subclasses")


class GeneratorModel(object):

  def __init__(self,
               src_reader: Optional['xnmt.models.InputReader'],
               trg_reader: Optional['xnmt.models.InputReader']):
    self.src_reader = src_reader
    self.trg_reader = trg_reader

  def generate(self, src: xnmt.Batch, search_strategy: 'xnmt.models.SearchStrategy', ref: Optional[xnmt.Batch]=None) \
      -> List[xnmt.structs.sentences.ReadableSentence]:
    hyps = self.create_trajectories(src, search_strategy, ref)
    return list(itertools.chain.from_iterable([self.hyp_to_readable(hyps[i], src[i].idx) for i in range(len(hyps))]))

  def create_trajectories(self, src: xnmt.Batch, search_strategy: 'xnmt.models.SearchStrategy', ref: Optional[xnmt.Batch]=None) \
      -> List[List[states.Hypothesis]]:
    outputs = []
    for i in range(src.batch_size()):
      src_i = xnmt.mark_as_batch(data=[src[i].get_unpadded_sent()], mask=None)
      if isinstance(search_strategy, xnmt.models.ForceableSearchStrategy) and search_strategy.is_forced():
        ref_i = xnmt.mark_as_batch(data=[ref[i].get_unpadded_sent()], mask=None)
        search_hyps = search_strategy.generate_forced_output(self, self.initial_state(src_i), ref_i)
      else:
        search_hyps = search_strategy.generate_output(self, self.initial_state(src_i))
      outputs.append(search_hyps)
    return outputs

  def hyp_to_readable(self, hyps: List[states.Hypothesis], idx: int) -> List[xnmt.structs.sentences.ReadableSentence]:
    raise NotImplementedError()

  def initial_state(self, src: xnmt.Batch):
    raise NotImplementedError()


class AutoRegressiveModel(ConditionedModel, GeneratorModel):
  def add_input(self, inp: xnmt.Batch, state: states.UniDirectionalState):
    raise NotImplementedError()

  def finish_generating(self, output: xnmt.Batch, dec_state: states.UniDirectionalState):
    raise NotImplementedError()

  def best_k(self, dec_state: states.UniDirectionalState, k: int, normalize_scores: bool) -> Sequence[states.SearchAction]:
    raise NotImplementedError()

  def sample(self, dec_state: states.UniDirectionalState, k: int) -> Sequence[states.SearchAction]:
    raise NotImplementedError()

  def pick_oracle(self, dec_state: states.UniDirectionalState, oracle):
    raise NotImplementedError()

  def hyp_to_readable(self, hyps: List[states.Hypothesis], idx: int):
    raise NotImplementedError()

  def initial_state(self, src: xnmt.Batch):
    raise NotImplementedError()

  def calc_nll(self, src: xnmt.Batch, trg: xnmt.Batch):
    raise NotImplementedError()
