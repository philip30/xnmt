"""
A module defining triggers to the common events used throughout XNMT.
"""


import xnmt


@xnmt.register_xnmt_event
def new_epoch(training_task, num_sents: int) -> None:
  """
  Trigger event indicating a new epoch for the specified task.

  Args:
    training_task: task that proceeds into next epoch.
    num_sents: number of training sentences in new epoch.
  """
  pass

@xnmt.register_xnmt_event
def set_train(val: bool) -> None:
  """
  Trigger event indicating enabling/disabling of "training" mode.

  Args:
    val: whether "training" mode is enabled
  """
  pass

@xnmt.register_xnmt_event
def start_sent(src_batch) -> None:
  """
  Trigger event indicating the start of a new sentence (or batch of sentences).

  Args:
    src: new sentence (or batch of sentences)
  """

@xnmt.register_xnmt_event_assign
def get_report_input(context) -> dict:
  return context


@xnmt.register_xnmt_event
def set_reporting(reporting: bool) -> None:
  pass

