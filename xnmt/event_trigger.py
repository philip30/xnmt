"""
A module defining triggers to the common events used throughout XNMT.
"""

import xnmt
import xnmt.models as models
import xnmt.globals as globals
import xnmt.internal.events as events


@events.register_xnmt_event
def new_epoch(training_task: models.TrainingTask, num_sents: int):
  """
  Trigger event indicating a new epoch for the specified task.

  Args:
    training_task: task that proceeds into next epoch.
    num_sents: number of training sentences in new epoch.
  """
  pass


@events.register_xnmt_event
def set_train(val: bool):
  """
  Trigger event indicating enabling/disabling of "training" mode.

  Args:
    val: whether "training" mode is enabled
  """
  globals.singleton_global.train = val


@events.register_xnmt_event
def start_sent(src_batch: xnmt.Batch):
  """
  Trigger event indicating the start of a new sentence (or batch of sentences).

  Args:
    src_batch: new sentence (or batch of sentences)
  """
  globals.singleton_global.src_batch = src_batch


@events.register_xnmt_event_assign
def get_report_input(context):
  pass


@events.register_xnmt_event
def set_reporting(val: bool):
  globals.singleton_global.reporting = val