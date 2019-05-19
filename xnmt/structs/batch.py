import dynet as dy
import numpy as np
import math
import collections

from typing import Optional, Any, Sequence

import xnmt.structs.sentences as sent


class Mask(object):
  """
  An immutable mask specifies padded parts in a sequence or batch of sequences.

  Masks are represented as numpy array of dimensions batchsize x seq_len, with parts
  belonging to the sequence set to 0, and parts that should be masked set to 1

  Args:
    np_arr: numpy array
  """
  def __init__(self, np_arr: np.ndarray) -> None:
    self.np_arr = np_arr
    self.np_arr.flags.writeable = False

  def __len__(self):
    return self.np_arr.shape[1]

  def batch_size(self) -> int:
    return self.np_arr.shape[0]

  def reversed(self) -> 'Mask':
    return Mask(self.np_arr[:,::-1])

  def add_to_tensor_expr(self, tensor_expr: dy.Expression, multiplicator: Optional[float]=None) -> dy.Expression:
    # TODO: might cache these expressions to save memory
    if np.count_nonzero(self.np_arr) == 0:
      return tensor_expr
    else:
      if multiplicator is not None:
        mask_expr = dy.inputTensor(np.expand_dims(self.np_arr.transpose(), axis=1) * multiplicator, batched=True)
      else:
        mask_expr = dy.inputTensor(np.expand_dims(self.np_arr.transpose(), axis=1), batched=True)
      return tensor_expr + mask_expr

  def lin_subsampled(self, reduce_factor: Optional[int] = None, trg_len: Optional[int]=None) -> 'Mask':
    if reduce_factor:
      return Mask(np.array([[self.np_arr[b,int(i*reduce_factor)] for i in range(int(math.ceil(len(self)/float(reduce_factor))))] for b in range(self.batch_size())]))
    else:
      return Mask(np.array([[self.np_arr[b,int(i*len(self)/float(trg_len))] for i in range(trg_len)] for b in range(self.batch_size())]))

  def cmult_by_timestep_expr(self, expr: dy.Expression, timestep: int, inverse: bool = False) -> dy.Expression:
    # TODO: might cache these expressions to save memory
    """
    Args:
      expr: a dynet expression corresponding to one timestep
      timestep: index of current timestep
      inverse: True will keep the unmasked parts, False will zero out the unmasked parts
    """
    if inverse:
      if np.count_nonzero(self.np_arr[:,timestep:timestep+1]) == 0:
        return expr
      mask_exp = dy.inputTensor((1.0 - self.np_arr)[:,timestep:timestep+1].transpose(), batched=True)
    else:
      if np.count_nonzero(self.np_arr[:,timestep:timestep+1]) == self.np_arr[:,timestep:timestep+1].size:
        return expr
      mask_exp = dy.inputTensor(self.np_arr[:,timestep:timestep+1].transpose(), batched=True)
    return dy.cmult(expr, mask_exp)


class Batch(collections.Sequence):
  """
  An abstract base class for minibatches of things.
  """
  def __init__(self, mask: Mask):
    self.mask = mask

  def batch_size(self) -> int: raise NotImplementedError()
  def sent_len(self) -> int: raise NotImplementedError()
  def __iter__(self): raise NotImplementedError()
  def __getitem__(self, item): raise NotImplementedError()
  def __len__(self): return self.batch_size()


class ListBatch(list, Batch):
  """
  A class containing a minibatch of things.

  This class behaves like a Python list, but adds semantics that the contents form a (mini)batch of things.
  An optional mask can be specified to indicate padded parts of the inputs.
  Should be treated as an immutable object.

  Args:
    batch_elements: list of things
    mask: optional mask when  batch contains items of unequal size
  """
  def __init__(self, batch_elements: collections.Sequence, mask: Mask=None) -> None:
    Batch.__init__(self, mask)
    assert len(batch_elements) > 0
    super().__init__(batch_elements)

  def batch_size(self) -> int:
    return super().__len__()

  def sent_len(self) -> int:
    return self[0].sent_len()

  def __getitem__(self, key):
    ret = super().__getitem__(key)
    if isinstance(key, slice):
      ret = ListBatch(ret)
    return ret


def mark_as_batch(data: Sequence, mask: Optional[Mask] = None) -> Batch:
  """
  Mark a sequence of items as batch

  Args:
    data: sequence of things
    mask: optional mask

  Returns: a batch of things
  """
  if isinstance(data, Batch) and mask is None:
    ret = data
  else:
    ret = ListBatch(data, mask)
  return ret


def is_batched(data: Any) -> bool:
  """
  Check whether some data is batched.

  Args:
    data: data to check

  Returns:
    True iff data is batched.
  """
  return isinstance(data, Batch)
