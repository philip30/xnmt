import typing
import numbers
import xnmt


class Scalar(xnmt.Serializable):
  yaml_tag = "!Scalar"
  """
  Scalar class for hyper parameter that support 1 value serialization.
  This class is actually a base class and does not have any different with simple python float/int.

  Args:
    initial: The value being hold by the scalar.
    times_updated: Is the epoch number.
  """
  @xnmt.serializable_init
  @xnmt.register_xnmt_handler
  def __init__(self, initial: numbers.Real = 0.0, times_updated: numbers.Integral = 0) -> None:
    self.initial = initial
    self.times_updated = times_updated
    self.value = self.get_curr_value()

  @xnmt.handle_xnmt_event
  def on_new_epoch(self, *args, **kwargs):
    self.value = self.get_curr_value()
    self.times_updated += 1
    self.save_processed_arg("times_updated", self.times_updated)

  def get_curr_value(self) -> numbers.Real:
    return self.initial

  def __repr__(self):
    return f"{self.__class__.__name__}[curr={self.get_curr_value()}]"

  # Operators
  def __lt__(self, b): return self.value < b
  def __le__(self, b): return self.value <= b
  def __eq__(self, b): return self.value == b
  def __ne__(self, b): return self.value != b
  def __ge__(self, b): return self.value >= b
  def __gt__(self, b): return self.value > b
  def __add__(self, b): return self.value + b
  def __sub__(self, b): return self.value - b
  def __mul__(self, b): return self.value * b
  def __neg__(self): return -self.value
  def __pos__(self): return +self.value
  def __pow__(self, b): return self.value ** b
  def __truediv__(self, b): return self.value / b
  def __floordiv__(self, b): return self.value // b

class DefinedSequence(Scalar, xnmt.Serializable):
  yaml_tag = "!DefinedSequence"
  """
  Class that represents a fixed defined sequence from config files.
  If update has been made more than the length of the sequence, the last element of the sequence will be returned instead

  x = DefinedSequence([0.1, 0.5, 1])

  # Epoch 1: 0.1
  # Epoch 2: 0.5
  # Epoch 3: 1
  # Epoch 4: 1
  # ...

  Args:
    sequence: A list of numbers
    times_updated: The epoch number
  """
  @xnmt.serializable_init
  def __init__(self, sequence: typing.Sequence[numbers.Real], times_updated: numbers.Integral = 0) -> None:
    self.sequence = sequence
    if len(sequence)==0: raise ValueError("DefinedSequence initialized with empty sequence")
    super().__init__(times_updated=times_updated)

  def get_curr_value(self) -> numbers.Real:
    return self.sequence[min(len(self.sequence) - 1, self.times_updated)]

numbers.Real.register(Scalar)
