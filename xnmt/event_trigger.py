"""
A module defining triggers to the common events used throughout XNMT.
"""

from typing import Union
import numbers
import random

from xnmt.train import tasks as training_tasks
from xnmt.models import base as models
from xnmt import batchers, events, losses, sent

@events.register_xnmt_event
def new_epoch(training_task: training_tasks.TrainingTask, num_sents: numbers.Integral) -> None:
  """
  Trigger event indicating a new epoch for the specified task.

  Args:
    training_task: task that proceeds into next epoch.
    num_sents: number of training sentences in new epoch.
  """
  pass

@events.register_xnmt_event
def set_train(val: bool) -> None:
  """
  Trigger event indicating enabling/disabling of "training" mode.

  Args:
    val: whether "training" mode is enabled
  """
  pass

@events.register_xnmt_event
def start_sent(src: Union[sent.Sentence, batchers.Batch]) -> None:
  """
  Trigger event indicating the start of a new sentence (or batch of sentences).

  Args:
    src: new sentence (or batch of sentences)
  """

@events.register_xnmt_event_assign
def get_report_input(context) -> dict:
  return context

@events.register_xnmt_event
def set_reporting(reporting: bool) -> None:
  pass

