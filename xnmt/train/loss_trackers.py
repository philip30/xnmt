import time
import numbers
import collections

from typing import Optional, Dict

import xnmt
import xnmt.models as models
import xnmt.internal.events as events
import xnmt.internal.utils as utils


class AccumTimeTracker(object):
  def __init__(self) -> None:
    self.start_time = None
    self.accum_time = 0.0

  def __enter__(self):
    self.start_time = time.time()

  def __exit__(self, *args):
    self.accum_time += time.time() - self.start_time

  def get_and_reset(self) -> numbers.Real:
    ret = self.accum_time
    self.accum_time = 0.0
    return ret


class TrainLossTracker(object):

  REPORT_TEMPLATE_SPEED = 'Epoch {epoch:.4f}: {data_name}_loss/word={loss:.6f} (words={words}, words/sec={words_per_sec:.2f}, time={time})'
  REPORT_TEMPLATE = 'Epoch {epoch:.4f}: {data_name}_loss/word={loss:.6f} (words={words}, words/sec={words_per_sec}, time={time})'
  REPORT_TEMPLATE_ADDITIONAL = '- {loss_name} {loss:5.6f}'
  REPORT_EVERY = 1

  @events.register_xnmt_handler
  def __init__(self, training_task: models.TrainingTask):
    self.training_task = training_task

    self.epoch_loss = collections.defaultdict(float)
    self.epoch_words = collections.defaultdict(int)
    self.last_report_sents_into_epoch = 0
    self.last_report_sents_since_start = 0

    self.last_report_words = 0

    self.time_tracker = AccumTimeTracker()
    self.start_time = time.time()
    self.name = self.training_task.name

  @events.handle_xnmt_event
  def on_new_epoch(self, training_task, num_sents):
    if training_task is self.training_task:
      self.epoch_loss.clear()
      self.epoch_words.clear()
      self.last_report_sents_since_start = 0
      self.last_report_words = 0

  def report(self, loss_data: Dict[str, int], trg: xnmt.Batch) -> None:
    """
    Accumulate training loss and report every REPORT_EVERY sentences.
    """
    for loss_name, (loss_value, units) in loss_data.items():
      self.epoch_words[loss_name] += units
      self.epoch_loss[loss_name] += loss_value

    self.epoch_words["__TRG__"] += sum(inp.len_unpadded() for inp in trg)

    sent_num_not_report = self.training_task.training_state.sents_since_start - self.last_report_sents_since_start
    should_report = sent_num_not_report >= TrainLossTracker.REPORT_EVERY \
                    or self.training_task.training_state.sents_into_epoch == self.training_task.cur_num_sentences()

    if should_report:
      fractional_epoch = (self.training_task.training_state.epoch_num - 1) \
                         + self.training_task.training_state.sents_into_epoch / self.training_task.cur_num_sentences()
      accum_time = self.time_tracker.get_and_reset()
      
      this_loss = sum([loss_value/units for loss_value, units in loss_data.values()])
      rep_train_loss = sum([self.epoch_loss[k]/self.epoch_words[k] for k in self.epoch_loss.keys()])
      utils.log_readable_and_tensorboard(
        template = TrainLossTracker.REPORT_TEMPLATE_SPEED if accum_time else TrainLossTracker.REPORT_TEMPLATE,
        args = {"loss": this_loss},
        n_iter = fractional_epoch,
        time = utils.format_time(time.time() - self.start_time),
        words = self.epoch_words["__TRG__"],
        data_name = "train",
        task_name = self.name,
        words_per_sec = (self.epoch_words["__TRG__"] - self.last_report_words) / accum_time if accum_time else None
      )

      if len(self.epoch_loss) > 1:
        for loss_name, (loss_values, units) in loss_data.items():
          utils.log_readable_and_tensorboard(template=TrainLossTracker.REPORT_TEMPLATE_ADDITIONAL,
                                             args={loss_name: loss_values / units},
                                             n_iter=fractional_epoch,
                                             data_name="train",
                                             task_name=self.name,
                                             loss_name=loss_name,
                                             loss=loss_values / self.epoch_words[loss_name])

      self.last_report_words = self.epoch_words["__TRG__"]
      self.last_report_sents_since_start = self.training_task.training_state.sents_since_start


class DevLossTracker(object):

  REPORT_TEMPLATE_DEV         = 'Epoch {epoch:.4f} dev {score} (time={time})'
  REPORT_TEMPLATE_DEV_AUX     = '             dev auxiliary {score}'
  REPORT_TEMPLATE_TIME_NEEDED = '             checkpoint took {time_needed}'

  def __init__(self,
               training_task: models.TrainingTask,
               eval_every: int,
               name: Optional[str]=None):
    self.training_task = training_task
    self.eval_dev_every = eval_every

    self.last_report_sents_since_start = 0
    self.fractional_epoch = 0

    self.dev_score = None
    self.aux_scores = []

    self.start_time = time.time()
    self.name = name
    self.time_tracker = AccumTimeTracker()

  def set_dev_score(self, dev_score: models.EvalScore):
    self.dev_score = dev_score

  def add_aux_score(self, score: models.EvalScore):
    self.aux_scores.append(score)

  def should_report_dev(self) -> bool:
    sent_num_not_report = self.training_task.training_state.sents_since_start - self.last_report_sents_since_start
    if self.eval_dev_every > 0:
      return sent_num_not_report >= self.eval_dev_every
    else:
      return sent_num_not_report >= self.training_task.cur_num_sentences()

  def report(self) -> None:
    this_report_time = time.time()
    self.last_report_sents_since_start = self.training_task.training_state.sents_since_start
    self.fractional_epoch = (self.training_task.training_state.epoch_num - 1) \
                            + self.training_task.training_state.sents_into_epoch / self.training_task.cur_num_sentences()
    dev_time = self.time_tracker.get_and_reset()
    utils.log_readable_and_tensorboard(template=DevLossTracker.REPORT_TEMPLATE_DEV,
                                       args={self.dev_score.metric_name(): self.dev_score.value()},
                                       n_iter=self.fractional_epoch,
                                       data_name="dev",
                                       task_name=self.name,
                                       score=self.dev_score,
                                       time=utils.format_time(this_report_time - self.start_time))
    for score in self.aux_scores:
      utils.log_readable_and_tensorboard(template=DevLossTracker.REPORT_TEMPLATE_DEV_AUX,
                                         args={score.metric_name(): score.value()},
                                         n_iter=self.fractional_epoch,
                                         data_name="dev",
                                         task_name=self.name,
                                         score=score)
    xnmt.logger.info(DevLossTracker.REPORT_TEMPLATE_TIME_NEEDED.format(time_needed= utils.format_time(dev_time),
                                                                  extra={"task_name" : self.name}))
    self.aux_scores = []

