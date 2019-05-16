import random
import asteval
import subprocess
import numpy as np

from typing import Iterator, Optional, Sequence, Union

import xnmt
import xnmt.structs.batchers as batchers
import xnmt.modules.input_readers as input_readers
import xnmt.models as models
import xnmt.train as train



class SimpleTrainingTask(models.TrainingTask, xnmt.Serializable):
  """
  Args:
    model: a trainable supervised model
    src_file: The file for the source data.
    trg_file: The file for the target data.
    dev_every: dev checkpoints every n sentences (0 for only after epoch)
    batcher: Type of batcher
    loss_calculator:
    run_for_epochs: number of epochs (None for unlimited epochs)
    lr_decay: decay learning rate by multiplying by this factor
    lr_decay_times:  Early stopping after decaying learning rate a certain number of times
    patience: apply LR decay after dev scores haven't improved over this many checkpoints
    initial_patience: if given, allows adjusting patience for the first LR decay
    dev_tasks: A list of tasks to run on the development set
    dev_combinator: A formula to combine together development scores into a single score to
                    choose whether to perform learning rate decay, etc.
                    e.g. 'x[0]-x[1]' would say that the first dev task score minus the
                    second dev task score is our measure of how good we're doing. If not
                    specified, only the score from the first dev task will be used.
    restart_trainer: Restart trainer (useful for Adam) and revert weights to best dev checkpoint when applying LR decay (https://arxiv.org/pdf/1706.09733.pdf)
    reload_command: Command to change the input data after each epoch.
                         --epoch EPOCH_NUM will be appended to the command.
                         To just reload the data after each epoch set the command to 'true'.
    sample_train_sents: If given, load a random subset of training sentences before each epoch. Useful when training data does not fit in memory.
    max_num_train_sents: Train only on the first n sentences
    max_src_len: Discard training sentences with source-side longer than this
    max_trg_len: Discard training sentences with target-side longer than this
    name: will be prepended to log outputs if given
  """
  @xnmt.serializable_init
  def __init__(self,
               model: models.ConditionedModel,
               src_file: Union[str, Sequence[str]] = None,
               trg_file: str = None,
               dev_every: int = 0,
               batcher: batchers.Batcher = xnmt.bare(batchers.SrcBatcher, batch_size=32),
               loss_calculator: models.LossCalculator = xnmt.bare(train.MLELoss),
               run_for_epochs: Optional[int] = None,
               lr_decay: float = 1.0,
               lr_decay_times: int = 3,
               patience: int = 1,
               initial_patience: Optional[int] = None,
               dev_tasks: Sequence[models.EvalTask] = None,
               dev_combinator = None,
               restart_trainer: bool = False,
               reload_command: Optional[str] = None,
               name: Optional[str] = None,
               sample_train_sents: Optional[int] = None,
               max_num_train_sents: Optional[int] = None,
               max_src_len: Optional[int] = None,
               max_trg_len: Optional[int] = None,
               src_reader: models.InputReader = xnmt.ref_src_reader,
               trg_reader: models.InputReader = xnmt.ref_trg_reader,
               trainer: models.XnmtOptimizer = xnmt.bare(xnmt.modules.optimizers.SGDTrainer)):
    super().__init__(model, models.TrainingState(), name, dev_every, run_for_epochs)
    self.src_file = src_file
    self.trg_file = trg_file
    self.dev_tasks = dev_tasks
    self.dev_combinator = dev_combinator
    self.trainer = trainer

    if lr_decay > 1.0 or lr_decay <= 0.0:
      raise RuntimeError("illegal lr_decay, must satisfy: 0.0 < lr_decay <= 1.0")
    self.lr_decay = lr_decay
    self.patience = patience
    self.initial_patience = initial_patience
    self.lr_decay_times = lr_decay_times
    self.restart_trainer = restart_trainer
    self.run_for_epochs = run_for_epochs

    self.early_stopping_reached = False
    # training state
    self.reload_command = reload_command

    self.model = model
    self.loss_calculator = loss_calculator

    self.sample_train_sents = sample_train_sents
    self.max_num_train_sents = max_num_train_sents
    self.max_src_len = max_src_len
    self.max_trg_len = max_trg_len

    self.batcher = batcher
    self.dev_loss_tracker = train.DevLossTracker(self, dev_every, name)
    self.name = name
    self.src_reader = src_reader
    self.trg_reader = trg_reader

  def _augment_data_initial(self):
    """
    Called before loading corpus for the first time, if reload_command is given
    """
    augment_command = self.reload_command
    xnmt.logger.debug('initial augmentation')
    if self._augmentation_handle is None:
      # first run
      self._augmentation_handle = subprocess.Popen(augment_command + " --epoch 0", shell=True)
      self._augmentation_handle.wait()

  def _augment_data_next_epoch(self):
    """
    This is run in the background if reload_command is given to prepare data for the next epoch
    """
    augment_command = self.reload_command
    if self._augmentation_handle is None:
      # first run
      self._augmentation_handle = subprocess.Popen(augment_command + " --epoch %d" % self.training_state.epoch_num, shell=True)
      self._augmentation_handle.wait()

    self._augmentation_handle.poll()
    retcode = self._augmentation_handle.returncode
    if retcode is not None:
      if self.training_state.epoch_num > 0:
        xnmt.logger.info('using reloaded data')
      # reload the data
      self.src_data, self.trg_data, self.src_batches, self.trg_batches = \
          input_readers.read_parallel_corpus(src_reader=self.src_reader,
                                             trg_reader=self.trg_reader,
                                             src_file=self.src_file,
                                             trg_file=self.trg_file,
                                             batcher=self.batcher,
                                             sample_sents=self.sample_train_sents,
                                             max_num_sents=self.max_num_train_sents,
                                             max_src_len=self.max_src_len,
                                             max_trg_len=self.max_trg_len)
      self.src_reader.train = self.trg_reader.train = False
      # restart data generation
      self._augmentation_handle = subprocess.Popen(augment_command + " --epoch %d" % self.training_state.epoch_num, shell=True)
    else:
      xnmt.logger.info('new data set is not ready yet, using data from last epoch.')

  def should_stop_training(self) -> bool:
    """
    Signal stopping if self.early_stopping_reached is marked or we exhausted the number of requested epochs.
    """
    return self.early_stopping_reached \
      or self.run_for_epochs is not None and (self.training_state.epoch_num > self.run_for_epochs
                                              or (self.training_state.epoch_num == self.run_for_epochs and
                                                  self.training_state.steps_into_epoch >= self.cur_num_minibatches()))

  def cur_num_minibatches(self) -> int:
    """
    Current number of minibatches (may change between epochs, e.g. for randomizing batchers or if reload_command is given)
    """
    return len(self.src_batches)

  def cur_num_sentences(self) -> int:
    """
    Current number of parallel sentences (may change between epochs, e.g. if reload_command is given)
    """
    return len(self.src_data)

  def _advance_epoch(self):
    """
    Shifts internal state to the next epoch, including data (re-)loading, batch re-packing and shuffling.
    """
    if self.reload_command is not None:
      if self.training_state.epoch_num==0:
        self._augmentation_handle = None
        self._augment_data_initial()
      else:
        self._augment_data_next_epoch()
    if self.training_state.epoch_num==0 or self.sample_train_sents or \
      self.src_reader.needs_reload() or self.trg_reader.needs_reload():
      xnmt.event_trigger.set_train(True)
      self.src_data, self.trg_data, self.src_batches, self.trg_batches = \
        input_readers.read_parallel_corpus(src_reader=self.src_reader, trg_reader=self.trg_reader,
                                           src_file=self.src_file, trg_file=self.trg_file,
                                           batcher=self.batcher, sample_sents=self.sample_train_sents,
                                           max_num_sents=self.max_num_train_sents,
                                           max_src_len=self.max_src_len, max_trg_len=self.max_trg_len)
      self.src_reader.train = self.trg_reader.train = False
    self.training_state.epoch_seed = random.randint(1,2147483647)
    random.seed(self.training_state.epoch_seed)
    np.random.seed(self.training_state.epoch_seed)
    self.src_batches, self.trg_batches = \
      self.batcher.pack(self.src_data, self.trg_data)
    self.training_state.epoch_num += 1
    self.training_state.steps_into_epoch = 0
    self.training_state.sents_into_epoch = 0
    self.minibatch_order = list(range(0, self.cur_num_minibatches()))
    np.random.shuffle(self.minibatch_order)
    xnmt.event_trigger.new_epoch(training_task=self, num_sents=self.cur_num_sentences())

  def next_minibatch(self) -> Iterator:
    """
    Infinitely loops over training minibatches and advances internal epoch state after every complete sweep over the corpus.

    Returns:
      Generator yielding (src_batch,trg_batch) tuples
    """
    while True:
      self._advance_epoch()
      for batch_num in self.minibatch_order:
        src = self.src_batches[batch_num]
        trg = self.trg_batches[batch_num]
        self.training_state.steps_into_epoch += 1
        self.training_state.sents_into_epoch += src.batch_size()
        self.training_state.sents_since_start += src.batch_size()
        yield src, trg

  def training_step(self, src: xnmt.Batch, trg: xnmt.Batch):
    """
    Perform forward pass for the next training step and handle training logic (switching epoch, reshuffling, ..)

    Args:
      src: src minibatch
      trg: trg minibatch
    Returns:
      Loss
    """
    return self.loss_calculator.calc_loss(self.model, src, trg)

  def checkpoint_needed(self):
    return self.dev_loss_tracker.should_report_dev()

  def checkpoint(self, control_learning_schedule: bool = True):
    """
    Performs a dev checkpoint

    Args:
      control_learning_schedule: If False, only evaluate dev data.
                                      If True, also perform model saving, LR decay etc. if needed.
    Returns:
      True if the model needs saving, False otherwise
    """
    # Perform evaluation
    if self.dev_tasks and len(self.dev_tasks) > 0:
      dev_scores = []
      with self.dev_loss_tracker.time_tracker:
        xnmt.logger.info(f"> Checkpoint [{self.name}]" if self.name else "> Checkpoint")
        for dev_task in self.dev_tasks:
          dev_score = dev_task.eval()
          if type(dev_score) == list:
            dev_scores.extend(dev_score)
          else:
            dev_scores.append(dev_score)
        self.dev_loss_tracker.set_dev_score(dev_scores[0])
        for dev_score in dev_scores[1:]:
          self.dev_loss_tracker.add_aux_score(dev_score)
      self.dev_loss_tracker.report()

      # Control the learning schedule
      if control_learning_schedule:
        # Check if this is the best
        is_best = False
        if self.dev_combinator is not None:
          x = [y.value() for y in dev_scores]
          aevala = asteval.Interpreter(symtable={'x': x})
          my_score = aevala(self.dev_combinator)
          xnmt.logger.info('  combined dev scores according to {}: {}'.format(self.dev_combinator, my_score))
          if self.training_state.best_dev_score is None or my_score > self.training_state.best_dev_score:
            self.training_state.best_dev_score = my_score
            is_best = True
        elif dev_scores[0].better_than(self.training_state.best_dev_score):
          self.training_state.best_dev_score = dev_scores[0]
          is_best = True
        # If this is the best, write the model out
        if is_best:
          self.training_state.cur_attempt = 0
          needs_saving = True
          xnmt.logger.info(f"  best dev score, writing out model")
        else:
          needs_saving = False
          # otherwise: learning rate decay / early stopping
          self.training_state.cur_attempt += 1
          if self.lr_decay < 1.0:
            should_decay = False
            if (self.initial_patience is None or self.training_state.num_times_lr_decayed>0) \
                    and self.training_state.cur_attempt >= self.patience:
              should_decay = True
            if self.initial_patience is not None and self.training_state.num_times_lr_decayed==0 \
                    and self.training_state.cur_attempt >= self.initial_patience:
              should_decay = True
            if should_decay:
              self.training_state.num_times_lr_decayed += 1
              if self.training_state.num_times_lr_decayed > self.lr_decay_times:
                xnmt.logger.info('  Early stopping')
                self.early_stopping_reached = True
              else:
                self.training_state.cur_attempt = 0
                self.trainer.learning_rate *= self.lr_decay
                xnmt.logger.info('  new learning rate: %s' % self.trainer.learning_rate)
                if self.restart_trainer:
                  xnmt.logger.info('  restarting trainer and reverting learned weights to best checkpoint..')
                  self.trainer.restart()
                  xnmt.internal.param_collections.ParamManager.param_col.revert_to_best_model()
      else: # case of not controling learning schedule
        needs_saving = False
    else: # case of no dev tasks
      needs_saving = True

    return needs_saving


