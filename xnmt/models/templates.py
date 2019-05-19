import dynet as dy
import numpy as np
import shutil
import itertools
import collections.abc as abc

import xnmt
import xnmt.models as models
import xnmt.structs.sentences as sent

from typing import List, Sequence, Iterator, Optional, Any, Callable, Union, Dict, Tuple


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
  def read_sents(self, filename: Union[str, List[str]], filter_ids: Sequence[int] = None) -> Iterator[xnmt.Sentence]:
    """
    Read sentences and return an iterator.

    Args:
      filename: data file
      filter_ids: only read sentences with these ids (0-indexed)
    Returns: iterator over sentences from filename
    """
    raise NotImplementedError()

  def read_sent(self, line: Any, idx: int):
    raise NotImplementedError()

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


class TrainingTask(object):
  """
  Base class for a training task. Training tasks can perform training steps
  and keep track of the training state, but may not implement the actual training
  loop.

  Args:
    model: The model to train
  """
  def __init__(self, model: models.TrainableModel, training_state: models.states.TrainingState, name: str,
               dev_every: int, run_for_epochs: int):
    self.model = model
    self.training_state = training_state
    self.name = name
    self.dev_every = dev_every
    self.run_for_epochs = run_for_epochs

  def should_stop_training(self) -> bool:
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
  def eval(self) -> Union[EvalScore, Sequence[EvalScore]]:
    raise NotImplementedError("EvalTask.eval() needs to be implemented in child classes")


class SearchStrategy(object):
  """
  A template class to generate translation from the output probability model. (Non-batched operation)
  """
  def generate_output(self,
                      generator: Union[models.GeneratorModel, models.AutoRegressiveModel],
                      initial_state: models.UniDirectionalState) -> List[models.Hypothesis]:
    """
    Args:
      generator: a generator
      initial_state: initial decoder state
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


class ReportInfo(object):
  """
  Info to pass to reporter

  Args:
    sent_info: list of dicts, one dict per sentence
    glob_info: a global dict applicable to each sentence
  """
  def __init__(self, sent_info: Optional[Sequence[Dict[str, Any]]] = None, glob_info: Optional[Dict[str, Any]] = None):
    if sent_info is None: sent_info = []
    if glob_info is None: glob_info = {}

    self.sent_info = sent_info
    self.glob_info = glob_info


class Reportable(object):
  """
  Base class for classes that contribute information to a report.

  Making an arbitrary class reportable requires to do the following:

  - specify ``Reportable`` as base class
  - call this super class's ``__init__()``, or do ``@register_xnmt_handler`` manually
  - pass either global info or per-sentence info or both:
    - call ``self.report_sent_info(d)`` for each sentence, where d is a dictionary containing info to pass on to the
      reporter
    - call ``self.report_corpus_info(d)`` once, where d is a dictionary containing info to pass on to the
      reporter
  """

  @xnmt.register_xnmt_handler
  def __init__(self) -> None:
    self._sent_info_list = []
    self._glob_info_list = {}
    self._is_reporting = False

  def report_sent_info(self, sent_info: Dict[str, Any]) -> None:
    """
    Add key/value pairs belonging to the current sentence for reporting.

    This should be called consistently for every sentence and in order.

    Args:
      sent_info: A dictionary of key/value pairs. The keys must match (be a subset of) the arguments in the reporter's
                 ``create_sent_report()`` method, and the values must be of the corresponding types.
    """
    if not hasattr(self, "_sent_info_list"):
      self._sent_info_list = []
    self._sent_info_list.append(sent_info)

  def report_corpus_info(self, glob_info: Dict[str, Any]) -> None:
    """
    Add key/value pairs for reporting that are relevant to all reported sentences.

    Args:
      glob_info: A dictionary of key/value pairs. The keys must match (be a subset of) the arguments in the reporter's
                 ``create_sent_report()`` method, and the values must be of the corresponding types.
    """
    if not hasattr(self, "_glob_info_list"):
      self._glob_info_list = {}
    self._glob_info_list.update(glob_info)

  @xnmt.handle_xnmt_event
  def on_get_report_input(self, context: ReportInfo) -> ReportInfo:
    if hasattr(self, "_glob_info_list"):
      context.glob_info.update(self._glob_info_list)
    if not hasattr(self, "_sent_info_list"):
      return context
    if len(context.sent_info)>0:
      assert len(context.sent_info) == len(self._sent_info_list), \
             "{} != {}".format(len(context.sent_info), len(self._sent_info_list))
    else:
      context.sent_info = []
      for _ in range(len(self._sent_info_list)): context.sent_info.append({})
    for context_i, sent_i in zip(context.sent_info, self._sent_info_list):
      context_i.update(sent_i)
    self._sent_info_list.clear()
    return context

  @xnmt.handle_xnmt_event
  def on_set_reporting(self, is_reporting: bool) -> None:
    self._sent_info_list = []
    self._is_reporting = is_reporting

  def is_reporting(self):
    return self._is_reporting if hasattr(self, "_is_reporting") else False


class Reporter(object):
  """
  A base class for a reporter that collects reportable information, formats it and writes it to disk.
  """
  def create_sent_report(self, **kwargs) -> None:
    """
    Create the report.

    The reporter should specify the arguments it needs explicitly, and should specify ``kwargs`` in addition to handle
    extra (unused) arguments without crashing.

    Args:
      **kwargs: additional arguments
    """
    raise NotImplementedError("must be implemented by subclasses")

  def conclude_report(self) -> None:
    raise NotImplementedError("must be implemented by subclasses")


class Inference(object):
  """
  A template class for classes that perform inference.

  Args:
    src_file: path of input src file to be translated
    trg_file: path of file where trg translatons will be written
    ref_file: path of file with reference translations, e.g. for forced decoding
    max_src_len: Remove sentences from data to decode that are longer than this on the source side
    max_num_sents: Stop decoding after the first n sentences.
    mode: type of decoding to perform.

            * ``onebest``: generate one best.
            * ``score``: output scores, useful for rescoring
            * ``forced``: perform forced decoding.
            * ``forceddebug``: perform forced decoding, calculate training loss, and make sure the scores are identical
              for debugging purposes.
    batcher: inference batcher, needed e.g. in connection with ``pad_src_token_to_multiple``
    reporter: a reporter to create reports for each decoded sentence
  """
  # TODO: Support k-best inference?
  def __init__(self,
               src_file: Optional[str] = None,
               trg_file: Optional[str] = None,
               ref_file: Optional[str] = None,
               max_src_len: Optional[int] = None,
               max_num_sents: Optional[int] = None,
               mode: str = "onebest",
               batcher: xnmt.structs.batchers.InOrderBatcher = xnmt.bare(xnmt.structs.batchers.InOrderBatcher, batch_size=1),
               reporter: Optional[Union[Reporter, Sequence[Reporter]]] = None,
               post_processor: Optional[Union[OutputProcessor, Sequence[OutputProcessor]]] = None):
    self.src_file = src_file
    self.trg_file = trg_file
    self.ref_file = ref_file
    self.max_src_len = max_src_len
    self.max_num_sents = max_num_sents
    self.mode = mode
    self.batcher = batcher
    self.reporter = reporter
    self.post_processor = post_processor

  def generate_one(
      self, generator: models.GeneratorModel, src: xnmt.Batch) -> Sequence[sent.ReadableSentence]:
    raise NotImplementedError("must be implemented by subclasses")

  def compute_losses_one(
      self, generator: models.GeneratorModel, src: xnmt.Batch, ref: xnmt.Batch) -> xnmt.FactoredLossExpr:
    raise NotImplementedError("must be implemented by subclasses")

  def perform_inference(
      self, generator: models.GeneratorModel, src_file: str = None, trg_file: str = None, ref_file: str = None):
    """
    Perform inference.

    Args:
      generator: the model to be used
      src_file: path of input src file to be translated
      trg_file: path of file where trg translatons will be written
      ref_file: ref file
    """
    src_file = src_file or self.src_file
    trg_file = trg_file or self.trg_file
    ref_file = ref_file or self.ref_file

    if trg_file is not None:
      xnmt.utils.make_parent_dir(trg_file)

    if ref_file is not None:
      xnmt.logger.info(f'Performing inference on {src_file} and {ref_file}')
    else:
      xnmt.logger.info(f'Performing inference on {src_file}')

    xnmt.event_trigger.set_train(False)

    ref_scores = None

    if self.mode in ['score', 'forceddebug']:
      ref_corpus, src_corpus = self._read_corpus(generator, src_file, mode=self.mode, ref_file=self.ref_file)
      ref_scores = self._compute_losses(generator, ref_corpus, src_corpus, self.max_num_sents)

    if self.mode == 'score':
      self._write_rescored_output(ref_scores, self.ref_file, trg_file)
    elif self.mode == 'forced' or self.mode == 'forceddebug':
      self._forced_decode(generator=generator, src_file=src_file,
                          ref_file=ref_file, batcher=self.batcher,
                          max_src_len=self.max_src_len,
                          assert_scores=ref_scores)
      if trg_file is not None:
        shutil.copyfile(ref_file, trg_file)
    else:
      self._generate_output(generator=generator,
                            src_file=src_file, trg_file=trg_file, batcher=self.batcher,
                            max_src_len=self.max_src_len)

  def _generate_one_batch(self,
                          generator: models.GeneratorModel,
                          src_batch: xnmt.Batch = None,
                          max_src_len: Optional[int] = None,
                          fp = None):
    """
    Generate outputs for a single batch and write them to the output file.
    """
    if max_src_len is not None and src_batch.sent_len() > max_src_len:
      output_txt = "\n".join([xnmt.globals.NO_DECODING_ATTEMPTED] * src_batch.batch_size())
      fp.write(f"{output_txt}\n")
    else:
      with xnmt.utils.ReportOnException({"src": src_batch, "graph": xnmt.utils.print_cg_conditional}):
        dy.renew_cg(immediate_compute=xnmt.settings.IMMEDIATE_COMPUTE, check_validity=xnmt.settings.CHECK_VALIDITY)
        outputs = self.generate_one(generator, src_batch)
        if self.reporter: self._create_sent_report()
        for i in range(len(outputs)):
          output_txt = outputs[i].sent_str(custom_output_procs=self.post_processor)
          fp.write(f"{output_txt}\n")

  def _generate_output(self,
                       generator: models.GeneratorModel,
                       src_file: str,
                       trg_file: str, batcher: Optional[xnmt.Batcher] = None,
                       max_src_len: Optional[int] = None) -> None:
    """
    Generate outputs and write them to file.

    Args:
      generator: generator model to use
      src_file: a file of src-side inputs to generate outputs for
      trg_file: file to write outputs to
      batcher: necessary with some cases of input pre-processing such as padding or truncation
      max_src_len: if given, skip inputs that are too long
    """
    src_in = generator.src_reader.read_sents(src_file)

    # Reporting is commenced if there is some defined reporters
    xnmt.event_trigger.set_reporting(self.reporter is not None)

    # Saving the translated output to a trg file
    with open(trg_file, 'wt', encoding='utf-8') as fp:
      src_batch = []
      for curr_sent_i, src_line in itertools.islice(enumerate(src_in), self.max_num_sents):
        src_batch.append(src_line)
        if len(src_batch) == batcher.batch_size:
          self._generate_one_batch(generator, xnmt.mark_as_batch(src_batch), max_src_len, fp)
          src_batch = []
      if len(src_batch) != 0:
        self._generate_one_batch(generator, xnmt.mark_as_batch(src_batch), max_src_len, fp)

    # Finishing up
    try:
      if xnmt.globals.is_reporting():
        self._conclude_report()
    finally:
      # Reporting is done in _generate_output only
      xnmt.event_trigger.set_reporting(False)

  def _forced_decode_one_batch(self, generator: models.GeneratorModel,
                               batcher: Optional[xnmt.Batcher] = None,
                               src_batch: xnmt.Batch = None,
                               ref_batch: xnmt.Batch = None,
                               assert_scores: xnmt.Batch = None,
                               max_src_len: Optional[int] = None):
    """
    Performs forced decoding for a single batch.
    """
    batch_size = len(src_batch)
    src_batches, ref_batches = batcher.pack(src_batch, ref_batch)
    src_batch = src_batches[0]
    ref_batch = ref_batches[0]
    src_len = src_batch.sent_len()

    # TODO(philip30): This if is nonsense
    if max_src_len is None or src_len <= max_src_len is not None and src_len > max_src_len:
      with xnmt.utils.ReportOnException({"src": src_batch, "graph": xnmt.utils.print_cg_conditional}):
        dy.renew_cg(immediate_compute=xnmt.settings.IMMEDIATE_COMPUTE, check_validity=xnmt.settings.CHECK_VALIDITY)
        outputs = self.generate_one(generator, src_batch)
        if self.reporter: self._create_sent_report()
        for i in range(len(outputs)):
          if assert_scores is not None:
            # If debugging forced decoding, make sure it matches
            assert batch_size == len(outputs), "debug forced decoding not supported with nbest inference"
            if (abs(outputs[i].score - assert_scores[i]) / abs(assert_scores[i])) > 1e-5:
              raise ValueError(
                f'Forced decoding score {outputs[i].score} and loss {assert_scores[i]} do not match at '
                f'sentence {i}')

  def _forced_decode(self,
                     generator: models.GeneratorModel,
                     src_file: str,
                     ref_file: str,
                     batcher: Optional[xnmt.Batcher] = None,
                     max_src_len: Optional[int] = None,
                     assert_scores: Optional[Sequence[float]] = None) -> None:
    """
    Perform forced decoding.

    Args:
      generator: generator model to use
      src_file: a file of src-side inputs to generate outputs for
      ref_file: path of file with reference translations
      batcher: necessary with some cases of input pre-processing such as padding or truncation
      max_src_len: if given, skip inputs that are too long
      assert_scores: if given, raise exception if the scores for generated outputs don't match the given scores
    """
    src_in = generator.src_reader.read_sents(src_file)

    # If we have a "assert scores" list return it, otherwise return "None" infinitely
    assert_in = assert_scores if assert_scores else iter(lambda: None, 1)

    # Reporting is commenced if there is some defined reporters
    xnmt.event_trigger.set_reporting(self.reporter is not None)

    # Saving the translated output to a trg file
    src_batch, ref_batch, assert_batch = [], [], []
    for curr_sent_i, (src_line, assert_line) in itertools.islice(enumerate(zip(src_in, assert_in)), self.max_num_sents):
      src_batch.append(src_line)
      assert_batch.append(assert_line)
      if len(src_batch) == batcher.batch_size:
        self._forced_decode_one_batch(
          generator, batcher, xnmt.mark_as_batch(src_batch), xnmt.mark_as_batch(assert_batch), max_src_len)
        src_batch, ref_batch, assert_batch = [], [], []
    if len(src_batch) != 0:
      self._forced_decode_one_batch(
        generator, batcher, xnmt.mark_as_batch(src_batch), xnmt.mark_as_batch(assert_batch), max_src_len)

    # Finishing up
    try:
      if xnmt.globals.is_reporting():
        self._conclude_report()
    finally:
      # Reporting is done in _generate_output only
      xnmt.event_trigger.set_reporting(False)

  def _create_sent_report(self):
    assert self.reporter is not None
    if not isinstance(self.reporter, abc.Iterable):
      self.reporter = [self.reporter]
    report_context = xnmt.event_trigger.get_report_input(context=models.ReportInfo())
    for report_input in report_context.sent_info:
      for reporter in self.reporter:
        reporter.create_sent_report(**report_input, **report_context.glob_info)

  def _conclude_report(self):
    assert self.reporter is not None
    if not isinstance(self.reporter, abc.Iterable):
      self.reporter = [self.reporter]
    for reporter in self.reporter:
      reporter.conclude_report()

  def _compute_losses(self, generator, ref_corpus, src_corpus, max_num_sents) -> List[float]:
    batched_src, batched_ref = self.batcher.pack(src_corpus, ref_corpus)
    ref_scores = []
    for sent_count, (src, ref) in enumerate(zip(batched_src, batched_ref)):
      if max_num_sents and sent_count >= max_num_sents: break
      dy.renew_cg(immediate_compute=xnmt.settings.IMMEDIATE_COMPUTE, check_validity=xnmt.settings.CHECK_VALIDITY)
      loss, _ = self.compute_losses_one(generator, src, ref).compute()
      if isinstance(loss.value(), abc.Iterable):
        ref_scores.extend(loss.value())
      else:
        ref_scores.append(loss.value())
    ref_scores = [-x for x in ref_scores]
    return ref_scores


  def _write_rescored_output(self, ref_scores: Sequence[float], ref_file: str, trg_file: str) :
    """
    Write scored sequences and scores to file when mode=='score'.

    Args:
      ref_scores: list of score values
      ref_file: filename where sequences to be scored a read from
      trg_file: filename to write to
    """
    with open(trg_file, 'wt', encoding='utf-8') as fp:
      with open(ref_file, "r", encoding="utf-8") as nbest_fp:
        for nbest, score in zip(nbest_fp, ref_scores):
          fp.write("{} ||| score={}\n".format(nbest.strip(), score))

  def _read_corpus(self, generator, src_file: str, mode: str, ref_file: str) -> Tuple[List, List]:
    src_corpus = list(generator.src_reader.read_sents(src_file))
    # Get reference if it exists and is necessary
    if mode == "forced" or mode == "forceddebug" or mode == "score":
      if ref_file is None:
        raise RuntimeError(f"When performing '{mode}' decoding, must specify reference file")
      score_src_corpus = []
      ref_corpus = []
      with open(ref_file, "r", encoding="utf-8") as fp:
        for idx, line in enumerate(fp):
          if mode == "score":
            nbest = line.split("|||")
            assert len(nbest) > 1, "When performing scoring, ref_file must have nbest format 'index ||| hypothesis'"
            src_index = int(nbest[0].strip())
            assert src_index < len(src_corpus), \
              f"The src_file has only {len(src_corpus)} instances, nbest file has invalid src_index {src_index}"
            score_src_corpus.append(src_corpus[src_index])
            trg_input = generator.trg_reader.read_sent(idx=idx, line=nbest[1].strip())
          else:
            trg_input = generator.trg_reader.read_sent(idx=idx, line=line)
          ref_corpus.append(trg_input)
      if mode == "score":
        src_corpus = score_src_corpus
    else:
      ref_corpus = None
    return ref_corpus, src_corpus


class Evaluator(object):
  """
  A template class to evaluate the quality of output.
  """

  def evaluate(self, ref: Sequence, hyp: Sequence, desc: Any = None) -> EvalScore:
    """
  Calculate the quality of output given a reference.

  Args:
    ref: list of reference sents ( a sentence is a list of tokens )
    hyp: list of hypothesis sents ( a sentence is a list of tokens )
    desc: optional description that is passed on to score objects
  Returns:
  """
    raise NotImplementedError('evaluate must be implemented in Evaluator subclasses')

  def evaluate_multi_ref(self, ref: Sequence[Sequence], hyp: Sequence, desc: Any = None) -> EvalScore:
    """
  Calculate the quality of output given multiple references.

  Args:
    ref: list of tuples of reference sentences ( a sentence is a list of tokens )
    hyp: list of hypothesis sentences ( a sentence is a list of tokens )
    desc: optional description that is passed on to score objects
  """
    raise NotImplementedError(f'evaluate_multi_ref() is not implemented for {type(self)}.')


class LengthNormalization(object):
  """
  A template class to adjust scores for length normalization during search.
  """

  def normalize_completed(self, completed_hyps: Sequence[models.Hypothesis],
                          src_length: Optional[int] = None) -> Sequence[float]:
    """
    Apply normalization step to completed hypotheses after search and return the normalized scores.

    Args:
      completed_hyps: list of completed Hypothesis objects, will be normalized in-place
      src_length: length of source sequence (None if not given)
    Returns:
      normalized scores
    """
    raise NotImplementedError('normalize_completed must be implemented in LengthNormalization subclasses')

  def normalize_partial_topk(self, score_so_far, score_to_add, new_len):
    """
    Apply normalization step after expanding a partial hypothesis and selecting the top k scores.

    Args:
      score_so_far: log score of the partial hypothesis
      score_to_add: log score of the top-k item that is to be added
      new_len: new length of partial hypothesis with current word already appended
    Returns:
      new score after applying score_to_add to score_so_far
    """
    return score_so_far + score_to_add # default behavior: add up the log probs

