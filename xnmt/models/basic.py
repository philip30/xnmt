import xnmt
import dynet as dy
import xnmt.models.states as states

from typing import Sequence, Optional


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
               src_reader: Optional['xnmt.models.InputReader'] = xnmt.ref_src_reader,
               trg_reader: Optional['xnmt.models.InputReader'] = xnmt.ref_trg_reader):
    self.src_reader = src_reader
    self.trg_reader = trg_reader

  def generate(self, src: xnmt.Batch, search_strategy: 'xnmt.models.SearchStrategy', is_sort=True) -> \
      Sequence[xnmt.structs.sentences.ReadableSentence]:
    pass

  def initial_state(self, src: xnmt.Batch) -> states.UniDirectionalState:
    raise NotImplementedError()

  def best_k(self, dec_state: states.DecoderState, k: int, normalize_scores: bool) -> Sequence[states.SearchAction]:
    raise NotImplementedError()

  def sample(self, dec_state: states.DecoderState, k: int) -> Sequence[states.SearchAction]:
    raise NotImplementedError()

class AutoRegressiveModel(object):
  def add_input(self, inp: xnmt.Batch, state: states.DecoderState):
    raise NotImplementedError()

  def finish_generating(self, output: xnmt.Batch, dec_state: states.DecoderState):
    raise NotImplementedError()
