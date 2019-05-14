import dynet as dy
import numpy as np

import xnmt
import xnmt.models as models

from typing import List, Sequence, Iterator, Optional, Any, Callable


class LossCalculator(object):
  """
  A template class implementing the training strategy and corresponding loss calculation.
  """
  def calc_loss(self,
                model: models.ConditionedModel,
                src: xnmt.Batch,
                trg: xnmt.Batch) -> xnmt.FactoredLossExpr:
    return self._perform_calc_loss(model, src, trg)

  def _perform_calc_loss(self,
                         model: models.ConditionedModel,
                         src: xnmt.Batch,
                         trg: xnmt.Batch) -> xnmt.FactoredLossExpr:
    raise NotImplementedError()


class InputReader(object):
  """
  A base class to read in a file and turn it into an input
  """
  def read_sents(self, filename: str, filter_ids: Sequence[int] = None) -> Iterator[xnmt.Sentence]:
    """
    Read sentences and return an iterator.

    Args:
      filename: data file
      filter_ids: only read sentences with these ids (0-indexed)
    Returns: iterator over sentences from filename
    """
    return self.iterate_filtered(filename, filter_ids)

  def count_sents(self, filename: str) -> int:
    """
    Count the number of sentences in a data file.

    Args:
      filename: data file
    Returns: number of sentences in the data file
    """
    raise RuntimeError("Input readers must implement the count_sents function")

  def needs_reload(self) -> bool:
    """
    Overwrite this method if data needs to be reload for each epoch
    """
    return False

  def iterate_filtered(self, filename: str, filter_ids:Optional[Sequence[int]]=None):
    raise NotImplementedError()


class TrainingTask(object):
  """
  Base class for a training task. Training tasks can perform training steps
  and keep track of the training state, but may not implement the actual training
  loop.

  Args:
    model: The model to train
  """
  def __init__(self, model: models.TrainableModel) -> None:
    self.model = model

  def should_stop_training(self):
    """
    Returns:
      True iff training is finished, i.e. training_step(...) should not be called again
    """
    raise NotImplementedError("must be implemented by subclasses")

  def training_step(self, **kwargs) -> xnmt.FactoredLossExpr:
    """
    Perform forward pass for the next training step and handle training logic (switching epoch, reshuffling, ..)

    Args:
      **kwargs: depends on subclass implementations
    Returns:
      Loss
    """
    raise NotImplementedError("must be implemented by subclasses")

  def next_minibatch(self) -> Iterator:
    """
    Infinitely loop over training minibatches.

    Returns:
      Generator yielding (src_batch,trg_batch) tuples
    """

  def checkpoint_needed(self) -> bool:
    raise NotImplementedError("must be implemented by subclasses")

  def checkpoint(self, control_learning_schedule: bool = False) -> bool:
    """
    Perform a dev checkpoint.

    Args:
      control_learning_schedule: If ``False``, only evaluate dev data.
                                 If ``True``, also perform model saving, LR decay etc. if needed.
    Returns:
      ``True`` iff the model needs saving
    """
    raise NotImplementedError("must be implemented by subclasses")

  def cur_num_minibatches(self) -> int:
    """
    Current number of minibatches (may change between epochs, e.g. for randomizing batchers or if reload_command is given)
    """
    raise NotImplementedError("must be implemented by subclasses")

  def cur_num_sentences(self) -> int:
    """
    Current number of parallel sentences (may change between epochs, e.g. if reload_command is given)
    """
    raise NotImplementedError("must be implemented by subclasses")


class EvalScore(object):
  """
  A template class for scores as resulting from using an :class:`Evaluator`.

  Args:
    desc: human-readable description to include in log outputs
  """
  def __init__(self, desc: Any = None) -> None:
    self.desc = desc

  def higher_is_better(self) -> bool:
    """
    Return ``True`` if higher values are favorable, ``False`` otherwise.

    Returns:
      Whether higher values are favorable.
    """
    raise NotImplementedError()

  def value(self) -> float:
    """
    Get the numeric value of the evaluated metric.

    Returns:
      Numeric evaluation score.
    """
    raise NotImplementedError()

  def metric_name(self) -> str:
    """
    Get the metric name.

    Returns:
      Metric name as string.
    """
    raise NotImplementedError()

  def score_str(self) -> str:
    """
    A string representation of the evaluated score, potentially including additional statistics.

    Returns:
      String representation of score.
    """
    raise NotImplementedError()

  def better_than(self, another_score: 'EvalScore') -> bool:
    """
    Compare score against another score and return ``True`` iff this score is better.

    Args:
      another_score: score to _compare against.

    Returns:
      Whether this score is better than ``another_score``.
    """
    if another_score is None or another_score.value() is None: return True
    elif self.value() is None: return False
    assert type(self) == type(another_score)
    if self.higher_is_better():
      return self.value() > another_score.value()
    else:
      return self.value() < another_score.value()

  def __str__(self):
    desc = getattr(self, "desc", None)
    if desc:
      return f"{self.metric_name()} ({desc}): {self.score_str()}"
    else:
      return f"{self.metric_name()}: {self.score_str()}"


class EvalTask(object):
  """
  An EvalTask is a task that does evaluation and returns one or more EvalScore objects.
  """
  def eval(self) -> EvalScore:
    raise NotImplementedError("EvalTask.eval() needs to be implemented in child classes")


class SearchStrategy(object):
  """
  A template class to generate translation from the output probability model. (Non-batched operation)
  """
  def generate_output(self,
                      generator: models.GeneratorModel,
                      initial_state: models.DecoderState,
                      src_length: Optional[int] = None) -> List[models.SearchAction]:
    """
    Args:
      generator: a generator
      initial_state: initial decoder state
      src_length: length of src sequence, required for some types of length normalization
    Returns:
      List of (word_ids, attentions, score, logsoftmaxes)
    """
    raise NotImplementedError('generate_output must be implemented in SearchStrategy subclasses')


class XnmtOptimizer(object):
  """
  A base classe for trainers. Trainers are mostly simple wrappers of DyNet trainers but can add extra functionality.

  Args:
    optimizer: the underlying DyNet optimizer (trainer)
    skip_noisy: keep track of a moving average and a moving standard deviation of the log of the gradient norm
                          values, and abort a step if the norm of the gradient exceeds four standard deviations of the
                          moving average. Reference: https://arxiv.org/pdf/1804.09849.pdf
  """

  def __init__(self, optimizer: dy.Trainer, skip_noisy: bool = False) -> None:
    self.optimizer = optimizer
    self.skip_noisy = skip_noisy
    if skip_noisy:
      self.rolling_stats = xnmt.utils.RollingStatistic()

  def update(self) -> None:
    """
    Update the parameters.
    """
    try:
      if not (self.skip_noisy and self._check_gradients_noisy()):
        self.optimizer.update()
      else:
        xnmt.logger.info("skipping noisy update")
    except RuntimeError:
      xnmt.logger.warning("Failed to perform update. Skipping example and clearing gradients.")
      for subcol in xnmt.internal.param_collections.ParamManager.param_col.subcols.values():
        for param in subcol.parameters_list():
          param.scale_gradient(0)

  def status(self) -> None:
    """
    Outputs information about the trainer in the stderr.

    (number of updates since last call, number of clipped gradients, learning rate, etc…)
    """
    return self.optimizer.status()

  def set_clip_threshold(self, thr: float) -> None:
    """
    Set clipping thershold

    To deactivate clipping, set the threshold to be <=0

    Args:
      thr: Clipping threshold
    """
    return self.optimizer.set_clip_threshold(thr)

  def get_clip_threshold(self) -> float:
    """
    Get clipping threshold

    Returns:
      Gradient clipping threshold
    """
    return self.optimizer.get_clip_threshold()

  def restart(self) -> None:
    """
    Restarts the optimizer

    Clears all momentum values and assimilate (if applicable)
    """
    return self.optimizer.restart()

  @property
  def learning_rate(self):
    return self.optimizer.learning_rate

  @learning_rate.setter
  def learning_rate(self, value):
    self.optimizer.learning_rate = value

  def _check_gradients_noisy(self) -> bool:
    sq_norm = 0
    for subcol in xnmt.internal.param_collections.ParamManager.param_col.subcols.values():
      for param in subcol.parameters_list():
        cur_grads = param.grad_as_array()
        sq_norm += np.sum(np.square(cur_grads))
    log_norm = np.log(np.sqrt(sq_norm))
    self.rolling_stats.update(log_norm)
    if self.rolling_stats.average is None: # too few statistics
      return False
    else:
      req_min = self.rolling_stats.average - 4*self.rolling_stats.stddev
      req_max = self.rolling_stats.average + 4*self.rolling_stats.stddev
      return not (req_min < log_norm < req_max)


class TrainingRegimen(object):
  """
  A training regimen is a class that implements a training loop.
  """
  def run_training(self, save_fct: Callable) -> None:
    """
    Run training steps in a loop until stopping criterion is reached.

    Args:
      save_fct: function to be invoked to save a model at dev checkpoints
    """
    raise NotImplementedError("")

  def backward(self, loss: dy.Expression, dynet_profiling: int) -> None:
    """
    Perform backward pass to accumulate gradients.

    Args:
      loss: Result of self.training_step(...)
      dynet_profiling: if > 0, print the computation graph
    """
    if dynet_profiling and dynet_profiling > 0:
      dy.print_text_graphviz()
    loss.backward()

  def update(self, trainer: XnmtOptimizer) -> None:
    """
    Update DyNet weights using the given optimizer.

    Args:
      trainer: DyNet trainer
    """
    trainer.update()


class ParamInitializer(object):
  """
  A parameter initializer that delegates to the DyNet initializers and possibly
  performs some extra configuration.
  """

  def initializer(self, dim, is_lookup: bool = False, num_shared: int = 1) -> 'dy.Initializer':
    """
    Args:
      dim: dimension of parameter tensor
      is_lookup: True if parameters are a lookup matrix
      num_shared: Indicates if one parameter object holds multiple matrices
    Returns:
      a dynet initializer object
    """
    raise NotImplementedError("subclasses must implement initializer()")


class OutputProcessor(object):
  def process(self, s: str) -> str:
    """
    Produce a string-representation of an output.

    Args:
      s: string to be processed

    Returns:
      post-processed string
    """
    raise NotImplementedError("must be implemented by subclasses")
