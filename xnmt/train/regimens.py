import contextlib
import numpy as np
import dynet as dy

from typing import Callable, Dict, Optional, Sequence, Union
from collections import OrderedDict

import xnmt
import xnmt.models as models
import xnmt.train as train


class SimpleTrainingRegimen(train.SimpleTrainingTask, models.TrainingRegimen, xnmt.Serializable):
  """
  Args:
    model: the model
    src_file: the source training file
    trg_file: the target training file
    dev_every: dev checkpoints every n sentences (0 for only after epoch)
    dev_zero: if True, add a checkpoint before training loop is entered (useful with pretrained networks).
    batcher: Type of batcher
    loss_calculator: The method for calculating the loss.
    trainer: Trainer object, default is SGD with learning rate 0.1
    run_for_epochs:
    lr_decay:
    lr_decay_times:  Early stopping after decaying learning rate a certain number of times
    patience: apply LR decay after dev scores haven't improved over this many checkpoints
    initial_patience: if given, allows adjusting patience for the first LR decay
    dev_tasks: A list of tasks to use during the development stage.
    dev_combinator: A formula to combine together development scores into a single score to
                    choose whether to perform learning rate decay, etc.
                    e.g. 'x[0]-x[1]' would say that the first dev task score minus the
                    second dev task score is our measure of how well we're doing. If not
                    specified, only the score from the first dev task will be used.
    restart_trainer: Restart trainer (useful for Adam) and revert weights to best dev checkpoint when applying
                            LR decay (https://arxiv.org/pdf/1706.09733.pdf)
    reload_command: Command to change the input data after each epoch.
                         --epoch EPOCH_NUM will be appended to the command.
                         To just reload the data after each epoch set the command to ``True``.
    name: will be prepended to log outputs if given
    sample_train_sents:
    max_num_train_sents:
    max_src_len:
    max_trg_len:
    loss_comb_method: method for combining loss across batch elements (``sum`` or ``avg``).
    update_every: simulate large-batch training by accumulating gradients over several steps before updating parameters
    commandline_args:
  """
  @xnmt.serializable_init
  def __init__(self,
               model: models.ConditionedModel = xnmt.Ref("model"),
               src_file: Union[None, str, Sequence[str]] = None,
               trg_file: Optional[str] = None,
               dev_every: int = 0,
               dev_zero: bool = False,
               batcher: xnmt.structs.batchers.Batcher = xnmt.bare(xnmt.structs.batchers.SrcBatcher, batch_size=32),
               loss_calculator: models.LossCalculator = xnmt.bare(train.MLELoss),
               trainer: models.XnmtOptimizer = xnmt.bare(xnmt.modules.optimizers.SGDTrainer, e0=0.1),
               run_for_epochs: Optional[int] = None,
               lr_decay: float= 1.0,
               lr_decay_times: int = 3,
               patience: int = 1,
               initial_patience: Optional[int] = None,
               dev_tasks: Sequence[models.EvalTask] = None,
               dev_combinator: Optional[str] = None,
               restart_trainer: bool = False,
               reload_command: Optional[str] = None,
               name: str = "{EXP}",
               sample_train_sents: Optional[int] = None,
               max_num_train_sents: Optional[int] = None,
               max_src_len: Optional[int] = None,
               max_trg_len: Optional[int] = None,
               loss_comb_method: str = xnmt.Ref("exp_global.loss_comb_method", default="sum"),
               update_every: int = 1,
               commandline_args: dict = xnmt.Ref("exp_global.commandline_args", default={})):

    super().__init__(model=model,
                     src_file=src_file,
                     trg_file=trg_file,
                     dev_every=dev_every,
                     batcher=batcher,
                     loss_calculator=loss_calculator,
                     run_for_epochs=run_for_epochs,
                     lr_decay=lr_decay,
                     lr_decay_times=lr_decay_times,
                     patience=patience,
                     initial_patience=initial_patience,
                     dev_tasks=dev_tasks,
                     dev_combinator=dev_combinator,
                     restart_trainer=restart_trainer,
                     reload_command=reload_command,
                     name=name,
                     sample_train_sents=sample_train_sents,
                     max_num_train_sents=max_num_train_sents,
                     max_src_len=max_src_len,
                     max_trg_len=max_trg_len,
                     trainer=trainer)
    self.dev_zero = dev_zero
    self.trainer = trainer or xnmt.modules.optimizers.SGDTrainer(e0=0.1)
    self.dynet_profiling = commandline_args.get("dynet_profiling", 0) if commandline_args else 0
    self.train_loss_tracker = train.TrainLossTracker(self)
    self.loss_comb_method = loss_comb_method
    self.update_every = update_every
    self.num_updates_skipped = 0

  def run_training(self, save_fct: Callable) -> None:
    """
    Main training loop (overwrites TrainingRegimen.run_training())
    """
    if self.run_for_epochs is None or self.run_for_epochs > 0:
      for src, trg in self.next_minibatch():
        if self.dev_zero:
          self.checkpoint_and_save(save_fct)
          self.dev_zero = False
        with xnmt.utils.ReportOnException({"src": src, "trg": trg, "graph": xnmt.utils.print_cg_conditional}):
          dy.renew_cg(immediate_compute=xnmt.settings.IMMEDIATE_COMPUTE, check_validity=xnmt.settings.CHECK_VALIDITY)
          with self.train_loss_tracker.time_tracker:
            xnmt.event_trigger.set_train(True)
            loss_expr = self.training_step(src, trg)
            loss, loss_value = loss_expr.compute()
            self.backward(loss, self.dynet_profiling)
            self.update(self.trainer)
          self.train_loss_tracker.report(loss_value, trg)
        if self.checkpoint_needed():
          self.checkpoint_and_save(save_fct)
        if self.should_stop_training(): break

  def update(self, trainer: models.XnmtOptimizer) -> None:
    self.num_updates_skipped += 1
    if self.num_updates_skipped == self.update_every:
      trainer.update()
      self.num_updates_skipped = 0
    else:
      assert 0 < self.num_updates_skipped < self.update_every

  def checkpoint_and_save(self, save_fct: Callable) -> None:
    should_save = self.checkpoint()
    if should_save:
      save_fct()


#class AutobatchTrainingRegimen(SimpleTrainingRegimen):
#  """
#  This regimen overrides SimpleTrainingRegimen by accumulating (summing) losses
#  into a FactoreLossExpr *before* running forward/backward in the computation graph.
#  It is designed to work with DyNet autobatching and when parts of architecture make
#  batching difficult (such as structured encoders like TreeLSTMS or Graph Networks).
#  The actual batch size is set through the "update_every" parameter, while the
#  underlying Batcher is expected to have "batch_size" equal to 1.
#
#  Args:
#    model: the model
#    src_file: the source training file
#    trg_file: the target training file
#    dev_every: dev checkpoints every n sentences (0 for only after epoch)
#    dev_zero: if True, add a checkpoint before training loop is entered (useful with pretrained networks).
#    batcher: Type of batcher
#    loss_calculator: The method for calculating the loss.
#    trainer: Trainer object, default is SGD with learning rate 0.1
#    run_for_epochs:
#    lr_decay:
#    lr_decay_times:  Early stopping after decaying learning rate a certain number of times
#    patience: apply LR decay after dev scores haven't improved over this many checkpoints
#    initial_patience: if given, allows adjusting patience for the first LR decay
#    dev_tasks: A list of tasks to use during the development stage.
#    dev_combinator: A formula to combine together development scores into a single score to
#                    choose whether to perform learning rate decay, etc.
#                    e.g. 'x[0]-x[1]' would say that the first dev task score minus the
#                    second dev task score is our measure of how good we're doing. If not
#                    specified, only the score from the first dev task will be used.
#    restart_trainer: Restart trainer (useful for Adam) and revert weights to best dev checkpoint when applying
#                            LR decay (https://arxiv.org/pdf/1706.09733.pdf)
#    reload_command: Command to change the input data after each epoch.
#                         --epoch EPOCH_NUM will be appended to the command.
#                         To just reload the data after each epoch set the command to ``True``.
#    name: will be prepended to log outputs if given
#    sample_train_sents:
#    max_num_train_sents:
#    max_src_len:
#    max_trg_len:
#    loss_comb_method: method for combining loss across batch elements (``sum`` or ``avg``).
#    update_every: how many instances to accumulate before updating parameters. This effectively sets the batch size under DyNet autobatching.
#    commandline_args:
#  """
#
#  @serializable_init
#  def __init__(self,
#               model: networks.ConditionedModel = Ref("model"),
#               src_file: Union[None, str, Sequence[str]] = None,
#               trg_file: Optional[str] = None,
#               dev_every: int = 0,
#               dev_zero: bool = False,
#               batcher: batchers.Batcher = bare(batchers.SrcBatcher, batch_size=32),
#               loss_calculator: loss_calculators.LossCalculator = bare(loss_calculators.MLELoss),
#               trainer: optimizers.XnmtOptimizer = bare(optimizers.SimpleSGDTrainer, e0=0.1),
#               run_for_epochs: Optional[int] = None,
#               lr_decay: float= 1.0,
#               lr_decay_times: int = 3,
#               patience: int = 1,
#               initial_patience: Optional[int] = None,
#               dev_tasks: Sequence[eval_tasks.EvalTask] = None,
#               dev_combinator: Optional[str] = None,
#               restart_trainer: bool = False,
#               reload_command: Optional[str] = None,
#               name: str = "{EXP}",
#               sample_train_sents: Optional[int] = None,
#               max_num_train_sents: Optional[int] = None,
#               max_src_len: Optional[int] = None,
#               max_trg_len: Optional[int] = None,
#               loss_comb_method: str = Ref("exp_global.loss_comb_method", default="sum"),
#               update_every: int = 1,
#               commandline_args: dict = Ref("exp_global.commandline_args", default={})) -> None:
#
#    super().__init__(model=model,
#                     src_file=src_file,
#                     trg_file=trg_file,
#                     dev_every=dev_every,
#                     batcher=batcher,
#                     loss_calculator=loss_calculator,
#                     run_for_epochs=run_for_epochs,
#                     lr_decay=lr_decay,
#                     lr_decay_times=lr_decay_times,
#                     patience=patience,
#                     initial_patience=initial_patience,
#                     dev_tasks=dev_tasks,
#                     dev_combinator=dev_combinator,
#                     restart_trainer=restart_trainer,
#                     reload_command=reload_command,
#                     name=name,
#                     sample_train_sents=sample_train_sents,
#                     max_num_train_sents=max_num_train_sents,
#                     max_src_len=max_src_len,
#                     max_trg_len=max_trg_len)
#    if batcher.batch_size != 1:
#      raise ValueError("AutobatchTrainingRegimen forces the batcher to have batch_size 1. Use update_every to set the actual batch size in this regimen.")
#    self.dev_zero = dev_zero
#    self.trainer = trainer or optimizers.SimpleSGDTrainer(e0=0.1)
#    self.dynet_profiling = commandline_args.get("dynet_profiling", 0) if commandline_args else 0
#    self.train_loss_tracker = loss_trackers.TrainLossTracker(self)
#    self.loss_comb_method = loss_comb_method
#    self.update_every = update_every
#    self.num_updates_skipped = 0
#
#  def run_training(self, save_fct: Callable) -> None:
#    """
#    Main training loop (overwrites TrainingRegimen.run_training())
#    """
#    dy.renew_cg(immediate_compute=settings.IMMEDIATE_COMPUTE, check_validity=settings.CHECK_VALIDITY)
#    if self.run_for_epochs is None or self.run_for_epochs > 0:
#      total_loss = []
#      # Needed for report
#      total_trg = []
#      for src, trg in self.next_minibatch():
#        if self.dev_zero:
#          self.checkpoint_and_save(save_fct)
#          self.dev_zero = False
#        with utils.ReportOnException({"src": src, "trg": trg, "graph": utils.print_cg_conditional}):
#          with self.train_loss_tracker.time_tracker:
#            event_trigger.set_train(True)
#            total_trg.append(trg[0])
#            total_loss.append(self.training_step(src, trg).compute())
#            # num_updates_skipped is incremented in update but
#            # we need to call backward before update
#            if self.num_updates_skipped == self.update_every - 1:
#              self.backward(sum([x[0] for x in total_loss]), self.dynet_profiling)
#            self.update(self.trainer)
#          if self.num_updates_skipped == 0:
#            total_loss_val = sum([x[1] for x in total_loss]) / len(total_loss)
#            reported_trg = batchers.ListBatch(total_trg)
#            self.train_loss_tracker.report(total_loss_val, reported_trg)
#            total_loss = []
#            total_trg = []
#            dy.renew_cg(immediate_compute=settings.IMMEDIATE_COMPUTE, check_validity=settings.CHECK_VALIDITY)
#        if self.checkpoint_needed():
#          # Do a last update before checkpoint
#          # Force forward-backward for the last batch even if it's smaller than update_every
#          self.num_updates_skipped = self.update_every - 1
#          self.backward(sum(x[0] for x in total_loss), self.dynet_profiling)
#          self.update(self.trainer)
#          total_loss_val = sum([x[1] for x in total_loss])
#          reported_trg = batchers.ListBatch(total_trg)
#          self.train_loss_tracker.report(total_loss_val, reported_trg)
#          total_loss = []
#          total_trg = []
#          self.checkpoint_and_save(save_fct)
#        if self.should_stop_training(): break


class MultiTaskTrainingRegimen(models.TrainingRegimen):
  """
  Base class for multi-task training classes.
  Mainly initializes tasks, performs sanity-checks, and manages set_train events.

  Args:
    tasks: list of training tasks.
                The first item takes on the role of the main task, meaning it
                will control early stopping, learning rate schedule, and
                model checkpoints.
    trainer: Trainer object, default is SGD with learning rate 0.1
    dev_zero: if True, add a checkpoint before training loop is entered (useful with pretrained networks).
    update_every: simulate large-batch training by accumulating gradients over several steps before updating parameters
    commandline_args:
  """
  def __init__(self,
               tasks: Sequence[models.TrainingTask],
               trainer: models.XnmtOptimizer = xnmt.bare(xnmt.modules.optimizers.SGDTrainer, e0=0.1),
               dev_zero: bool = False,
               update_every: int = 1,
               commandline_args: dict = xnmt.Ref("exp_global.commandline_args", default=None)):
    super().__init__()
    self.dynet_profiling = commandline_args.get("dynet_profiling", 0) if commandline_args else 0
    if len(tasks)==0: raise ValueError("Task list must be non-empty.")
    self.tasks = tasks
    self.trainer = trainer
    for task in tasks[1:]:
      if hasattr(task, "trainer") and task.trainer is not None:
        raise ValueError("Can instantiate only one trainer object. "
                         "Possibly, multiple training regimens were created when training tasks should have been used.")
    self.train = None
    self.model_file = xnmt.internal.param_collections.ParamManager.param_col.model_file
    for task in tasks:
      task.trainer = trainer
    self.dev_zero = dev_zero
    self.update_every = update_every
    self.num_updates_skipped = 0

  def trigger_train_event(self, value: bool) -> None:
    """
    Trigger set_train event, but only if that would lead to a change of the value
    of set_train.
    Args:
      value: True or False
    """
    if self.train is None:
      self.train = value
      xnmt.event_trigger.set_train(value)
    else:
      if value!=self.train:
        self.train = value
        xnmt.event_trigger.set_train(value)

  def update(self, trainer: models.XnmtOptimizer) -> None:
    self.num_updates_skipped += 1
    if self.num_updates_skipped == self.update_every:
      trainer.update()
      self.num_updates_skipped = 0
    else:
      assert 0 < self.num_updates_skipped < self.update_every


class SameBatchMultiTaskTrainingRegimen(MultiTaskTrainingRegimen, xnmt.Serializable):
  """
  Multi-task training where gradients are accumulated and weight updates are thus performed jointly for each task.
  The relative weight between tasks can be configured setting the number of steps to accumulate over for each task.
  Note that the batch size for each task also has an influence on task weighting.
  The stopping criterion of the first task is used (other tasks' stopping criteria are ignored).

  Args:
    tasks: Training tasks
    trainer: The trainer is shared across tasks
    dev_zero: If ``True``, add a checkpoint before training loop is entered (useful with pretrained networks).
    per_task_backward: If ``True``, call backward() for each task separately and renew computation graph between
                       tasks. Yields the same results, but ``True`` uses less memory while ``False`` may be
                       faster when using autobatching.
    loss_comb_method: Method for combining loss across batch elements ('sum' or 'avg').
    update_every: Simulate large-batch training by accumulating gradients over several steps before updating parameters.
                  This is implemented as an outer loop, i.e. we first accumulate gradients from steps for each task,
                  and then loop according to this parameter so that we collect multiple steps for each task and always
                  according to the same ratio.
    n_task_steps: The number steps to accumulate for each task, useful for weighting tasks.
    commandline_args:
  """
  @xnmt.serializable_init
  def __init__(self,
               tasks: Sequence[models.TrainingTask],
               trainer: models.XnmtOptimizer = xnmt.bare(xnmt.modules.optimizers.SGDTrainer, e0=0.1),
               dev_zero: bool = False,
               per_task_backward: bool = True,
               loss_comb_method: str = xnmt.Ref("exp_global.loss_comb_method", default="sum"),
               update_every: int = 1,
               n_task_steps: Optional[Sequence[int]] = None,
               commandline_args: dict = xnmt.Ref("exp_global.commandline_args", default=None)) -> None:
    super().__init__(tasks=tasks, trainer=trainer, dev_zero=dev_zero, update_every=update_every,
                     commandline_args=commandline_args)
    self.train_loss_trackers = {task : xnmt.train.TrainLossTracker(task) for task in tasks}
    self.per_task_backward = per_task_backward
    self.loss_comb_method = loss_comb_method
    self.n_task_steps = n_task_steps or [1] * len(tasks)
    if len(self.n_task_steps) != len(tasks):
      raise ValueError(f"number of tasks and steps per task do not match: {len(tasks)} != {len(self.n_task_steps)}")

  def run_training(self, save_fct: Callable) -> None:
    task_generators = OrderedDict()
    for task in self.tasks:
      task_generators[task] = task.next_minibatch()
    if self.tasks[0].run_for_epochs > 0:
      while True:
        task_losses = []
        task_src_trg = []
        for (task, task_gen), task_n in zip(task_generators.items(), self.n_task_steps):
          for _ in range(task_n):
            src, trg = next(task_gen)
            task_src_trg.append((task, src, trg))
        if self.dev_zero: # True only in first iteration
          self.checkpoint_and_save(save_fct)
        dy.renew_cg(immediate_compute=xnmt.settings.IMMEDIATE_COMPUTE, check_validity=xnmt.settings.CHECK_VALIDITY)
        task_trg_loss_stats = {}
        with contextlib.ExitStack() as stack: #use exit stack to control whether to use global or per-task time tracking
          if not self.per_task_backward:
            stack.enter_context(self.train_loss_trackers[self.tasks[0]].time_tracker)
          self.trigger_train_event(True)
          for task, src, trg in task_src_trg:
            with contextlib.ExitStack() as stack2:
              if self.per_task_backward:
                stack2.enter_context(self.train_loss_trackers[task].time_tracker)
              loss_expr, loss_value = task.training_step(src, trg).compute()
              task_trg_loss_stats[task] = (trg, loss_value)
              if self.per_task_backward:
                self.backward(loss_expr, self.dynet_profiling)
                dy.renew_cg(immediate_compute=xnmt.settings.IMMEDIATE_COMPUTE, check_validity=xnmt.settings.CHECK_VALIDITY)
              else:
                task_losses.append(loss_expr)
          if not self.per_task_backward:
            self.backward(sum(task_losses), self.dynet_profiling)
          self.update(self.trainer)
        for task, (trg, stats) in task_trg_loss_stats.items():
          self.train_loss_trackers[task].report(stats, trg)
        self.checkpoint_and_save(save_fct)
        if self.tasks[0].should_stop_training(): break

  def checkpoint_and_save(self, save_fct: Callable) -> None:
    for task_i, task in enumerate(self.tasks):
      if self.dev_zero or task.checkpoint_needed():
        should_save = task.checkpoint(control_learning_schedule=(task_i == 0))
        if should_save:
          save_fct()
    self.dev_zero = False



class AlternatingBatchMultiTaskTrainingRegimen(MultiTaskTrainingRegimen, xnmt.Serializable):
  """
  Multi-task training where training steps are performed one after another.

  The relative weight between tasks are explicitly specified explicitly, and for each step one task is drawn at random
  accordingly.
  The stopping criterion of the first task is used (other tasks' stopping criteria are ignored).

  Args:
    tasks: training tasks
    trainer: the trainer is shared across tasks
    dev_zero: if True, add a checkpoint before training loop is entered (useful with pretrained networks).
    loss_comb_method: method for combining loss across batch elements ('sum' or 'avg').
    update_every_within: Simulate large-batch training by accumulating gradients over several steps before updating
                         parameters. The behavior here is to draw multiple times from the same task until update is
                         invoked.
    update_every_across: Simulate large-batch training by accumulating gradients over several steps before updating
                         parameters. The behavior here is to draw tasks randomly several times before doing parameter
                         updates.
    commandline_args:
  """
  @xnmt.serializable_init
  def __init__(self,
               tasks: Sequence[models.TrainingTask],
               task_weights: Optional[Sequence[float]] = None,
               trainer: models.XnmtOptimizer = xnmt.bare(xnmt.modules.optimizers.SGDTrainer, e0=0.1),
               dev_zero: bool = False,
               loss_comb_method: str = xnmt.Ref("exp_global.loss_comb_method", default="sum"),
               update_every_within: int = 1,
               update_every_across: int = 1,
               commandline_args: Optional[dict] = xnmt.Ref("exp_global.commandline_args", default=None)) -> None:
    super().__init__(tasks=tasks, trainer=trainer, dev_zero=dev_zero, update_every=update_every_across,
                     commandline_args=commandline_args)
    if update_every_within!=1 and update_every_across!=1:
      raise ValueError("update_every_within and update_every_across cannot be mixed.")
    self.update_every_within = update_every_within
    self.task_weights = task_weights or [1./len(tasks)] * len(tasks)
    if len(self.task_weights) != len(self.tasks):
      raise ValueError(f"number of tasks must match number of task weights; "
                       f"found: {len(self.task_weights)} != {len(self.tasks)}")
    self.train_loss_trackers = {task: xnmt.train.TrainLossTracker(task) for task in tasks}
    self.loss_comb_method = loss_comb_method

  def run_training(self, save_fct: Callable):
    task_generators = OrderedDict()
    for task in self.tasks:
      task_generators[task] = task.next_minibatch()
    dev_zero = {i:self.dev_zero for i in range(len(self.tasks))}
    if self.tasks[0].run_for_epochs > 0:
      while True:
        dy.renew_cg(immediate_compute=xnmt.settings.IMMEDIATE_COMPUTE, check_validity=xnmt.settings.CHECK_VALIDITY)
        cur_task_i = np.random.choice(range(len(self.tasks)), p=self.task_weights)
        cur_task = self.tasks[cur_task_i]
        task_gen = task_generators[cur_task]
        if dev_zero[cur_task_i]: self.checkpoint_and_save(cur_task, cur_task_i, save_fct, dev_zero)
        cur_train_loss_tracker = self.train_loss_trackers[cur_task]
        with cur_train_loss_tracker.time_tracker:
          for _ in range(self.update_every_within):
            src, trg = next(task_gen)
            self.trigger_train_event(True)
            loss_expr, loss_value = cur_task.training_step(src, trg).compute()
            self.backward(loss=loss_expr, dynet_profiling=self.dynet_profiling)
            cur_train_loss_tracker.report(loss_value, trg)
          self.update(trainer=self.trainer)
         
        self.checkpoint_and_save(cur_task, cur_task_i, save_fct, dev_zero)
        if self.tasks[0].should_stop_training(): break

  def checkpoint_and_save(self,
                          cur_task: models.TrainingTask,
                          cur_task_i: int,
                          save_fct: Callable,
                          dev_zero: Dict[int,bool]):
    if dev_zero[cur_task_i] or cur_task.checkpoint_needed():
      dev_zero[cur_task_i] = False
      should_save = cur_task.checkpoint(control_learning_schedule=(cur_task_i == 0))
      if should_save:
        save_fct()


class SerialMultiTaskTrainingRegimen(MultiTaskTrainingRegimen, xnmt.Serializable):
  """
  Trains only first task until stopping criterion met, then the same for the second task, etc.

  Useful to realize a pretraining-finetuning strategy.

  Args:
    tasks: training tasks. The currently active task is treated as main task.
    trainer: the trainer is shared across tasks
    dev_zero: if True, add a checkpoint before training loop is entered (useful with pretrained networks).
    loss_comb_method: method for combining loss across batch elements ('sum' or 'avg').
    update_every: simulate large-batch training by accumulating gradients over several steps before updating parameters
    commandline_args:
  """

  @xnmt.serializable_init
  def __init__(self,
               tasks: Sequence[models.TrainingTask],
               trainer: models.XnmtOptimizer = xnmt.bare(xnmt.modules.optimizers.SGDTrainer, e0=0.1),
               dev_zero: bool = False,
               loss_comb_method: str = xnmt.Ref("exp_global.loss_comb_method", default="sum"),
               update_every: int = 1,
               commandline_args: Optional[dict] = xnmt.Ref("exp_global.commandline_args", default=None)):
    super().__init__(tasks=tasks, trainer=trainer, dev_zero=dev_zero, commandline_args=commandline_args,
                     update_every=update_every)
    self.train_loss_trackers = {task: xnmt.train.TrainLossTracker(task) for task in tasks}
    self.loss_comb_method = loss_comb_method

  def run_training(self, save_fct: Callable) -> None:
    dev_zero = {i:self.dev_zero for i in range(len(self.tasks))}
    for cur_task_id in range(len(self.tasks)):
      self.train = None
      cur_task = self.tasks[cur_task_id]
      cur_train_loss_tracker = self.train_loss_trackers[cur_task]
      task_gen = cur_task.next_minibatch()
      if cur_task.run_for_epochs > 0:
        while True:
          dy.renew_cg(immediate_compute=xnmt.settings.IMMEDIATE_COMPUTE, check_validity=xnmt.settings.CHECK_VALIDITY)
          src, trg = next(task_gen)
          if dev_zero[cur_task_id]: self.checkpoint_and_save(cur_task, cur_task_id, save_fct, dev_zero)
          with cur_train_loss_tracker.time_tracker:
            self.trigger_train_event(True)
            loss_builder = cur_task.training_step(src, trg)
            task_loss, task_loss_value = loss_builder.compute()
            self.backward(task_loss, self.dynet_profiling)
            self.update(self.trainer)
          cur_train_loss_tracker.report(task_loss_value, trg)
          self.checkpoint_and_save(cur_task, cur_task_id, save_fct, dev_zero)
          if cur_task.should_stop_training(): break

  def checkpoint_and_save(self,
                          cur_task: models.TrainingTask,
                          cur_task_id: int,
                          save_fct: Callable,
                          dev_zero: Dict[int,bool]) -> None:
    if dev_zero[cur_task_id] or cur_task.checkpoint_needed():
      dev_zero[cur_task_id] = False
      should_save = cur_task.checkpoint(control_learning_schedule=True)
      if should_save:
        save_fct()
