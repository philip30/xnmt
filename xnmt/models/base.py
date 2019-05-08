from typing import Optional, Sequence, Union

import numpy as np
import dynet as dy

import xnmt.events as events

from xnmt import batchers, input_readers, sent, vocabs
from xnmt.losses import LossExpr
from xnmt.persistence import Serializable, serializable_init

class TrainableModel(object):
  """
  A template class for a basic trainable model, implementing a loss function.
  """

  def calc_nll(self, *args, **kwargs) -> LossExpr:
    """Calculate loss based on input-output pairs.

    Losses are accumulated only across unmasked timesteps in each batch element.

    Arguments are to be defined by subclasses

    Returns:
      A (possibly batched) expression representing the loss.
    """

class UnconditionedModel(TrainableModel):
  """
  A template class for trainable model that computes target losses without conditioning on other inputs.

  Args:
    trg_reader: target reader
  """

  def __init__(self, trg_reader: input_readers.InputReader) -> None:
    self.trg_reader = trg_reader

  def calc_nll(self, trg: Union[batchers.Batch, sent.Sentence]) -> LossExpr:
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

  Args:
    src_reader: source reader
    trg_reader: target reader
  """

  def __init__(self, src_reader: input_readers.InputReader, trg_reader: input_readers.InputReader) -> None:
    self.src_reader = src_reader
    self.trg_reader = trg_reader

  def calc_nll(self, src: Union[batchers.Batch, sent.Sentence], trg: Union[batchers.Batch, sent.Sentence]) \
          -> LossExpr:
    """Calculate loss based on input-output pairs.

    Losses are accumulated only across unmasked timesteps in each batch element.

    Args:
      src: The source, a sentence or a batch of sentences.
      trg: The target, a sentence or a batch of sentences.

    Returns:
      A (possibly batched) expression representing the loss.
    """
    raise NotImplementedError("must be implemented by subclasses")


class PolicyConditionedModel(object):
  def __init__(self, policy_network, policy_train_oracle, policy_test_oracle):
    self.policy_network = policy_network
    self.policy_train_oracle = policy_train_oracle
    self.policy_test_oracle = policy_test_oracle
    # Model state
    self.actions = []
    self.train = True
    self.src_sents = None

  def calc_policy_nll(self, src, trg, parent_model):
    raise NotImplementedError("must be implemented by subclasses")

  def create_trajectories(self, *args, **kwargs):
    raise NotImplementedError("Must be implemented by subclasses")

  def create_trajectory(self, src, trg, current_state=None, from_oracle=False, force_decoding=True, parent_model=None):
    raise NotImplementedError("Must be implemented by subclasses")

  def reset_policy_states(self):
    self.actions.clear()

  @events.handle_xnmt_event
  def on_start_sent(self, src_batch):
    self.reset_policy_states()
    self.src_sents = src_batch

  @events.handle_xnmt_event
  def on_set_train(self, train):
    self.train = train


class GeneratorModel(object):
  """
  A template class for models that can perform inference to generate some kind of output.

  Args:
    src_reader: source input reader
    trg_reader: an optional target input reader, needed in some cases such as n-best scoring
  """
  def __init__(self, src_reader: input_readers.InputReader, trg_reader: Optional[input_readers.InputReader] = None) \
          -> None:
    self.src_reader = src_reader
    self.trg_reader = trg_reader

  def generate(self, src: batchers.Batch, *args, **kwargs) -> Sequence[sent.ReadableSentence]:
    """
    Generate outputs.

    Args:
      src: batch of source-side inputs
      *args:
      **kwargs: Further arguments to be specified by subclasses
    Returns:
      output objects
    """
    raise NotImplementedError("must be implemented by subclasses")

  def eog_symbol(self):
    """
    Specify the end of generation symbol.
    """
    return vocabs.Vocab.ES

  def finish_generating(self, output, dec_state):
    eog_symbol = self.eog_symbol()
    if type(output) == np.ndarray or type(output) == list:
      return [out_i == eog_symbol for out_i in output]
    else:
      return output == self.eog_symbol()


class CascadeGenerator(GeneratorModel, Serializable):
  """
  A cascade that chains several generator models.

  This generator does not support calling ``generate()`` directly. Instead, it's sub-generators should be accessed
  and used to generate outputs one by one.

  Args:
    generators: list of generators
  """
  yaml_tag = '!CascadeGenerator'

  @serializable_init
  def __init__(self, generators: Sequence[GeneratorModel]) -> None:
    super().__init__(src_reader = generators[0].src_reader, trg_reader = generators[-1].trg_reader)
    self.generators = generators

  def generate(self, *args, **kwargs) -> Sequence[sent.ReadableSentence]:
    raise ValueError("cannot call CascadeGenerator.generate() directly; access the sub-generators instead.")
