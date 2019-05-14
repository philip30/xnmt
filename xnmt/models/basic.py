import xnmt
import dynet as dy
import xnmt.models.states as states

from typing import Sequence


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

class UnconditionedModel(object):
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

class ConditionedModel(object):

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

  def __init__(self, search_output_processor, eog_symbol=xnmt.Vocab.ES):
    self.srch_out_prcs = search_output_processor
    self.eog_symbol = eog_symbol

  def generate(self, src: xnmt.Batch, search_strategy, is_sort=True) -> Sequence[Sequence[
    xnmt.structs.sentences.ReadableSentence]]:
    pass

  def initial_state(self, src: xnmt.Batch) -> states.UniDirectionalState:
    raise NotImplementedError()

  def add_input(self, inp: xnmt.Batch, state: states.DecoderState) -> states.UniDirectionalState:
    raise NotImplementedError()

  def finish_generating(self, output: xnmt.Batch, dec_state: states.DecoderState):
    raise NotImplementedError()

