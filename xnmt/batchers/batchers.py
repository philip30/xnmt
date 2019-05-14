import random
import numbers
import numpy as np
import dynet as dy
import collections
import xnmt

from typing import Callable, Optional, Sequence, Tuple, Union

import xnmt.structs.batch as batch
import xnmt.structs.sentences as sent


class Batcher(object):
  """
  A template class to convert a list of sentences to several batches of sentences.

  Args:
    batch_size: batch size
    granularity: 'sent' or 'word'
    pad_src_to_multiple: pad source sentences so its length is multiple of this integer.
    sort_within_by_trg_len: whether to sort by reverse trg len inside a batch
  """

  def __init__(self,
               batch_size: int,
               granularity: str = 'sent',
               pad_src_to_multiple: int = 1,
               sort_within_by_trg_len: bool = True) -> None:
    self.batch_size = batch_size
    self.granularity = granularity
    self.pad_src_to_multiple = pad_src_to_multiple
    self.sort_within_by_trg_len = sort_within_by_trg_len

  def is_random(self) -> bool:
    """
    Returns:
      True if there is some randomness in the batching process, False otherwise.
    """
    return False

  def create_single_batch(self,
                          src_sents: Sequence[sent.Sentence],
                          trg_sents: Optional[Sequence[sent.Sentence]] = None,
                          sort_by_trg_len: bool = False) -> Union[xnmt.Batch, Tuple[xnmt.Batch, xnmt.Batch]]:
    """
    Create a single batch, either source-only or source-and-target.

    Args:
      src_sents: list of source-side inputs
      trg_sents: optional list of target-side inputs
      sort_by_trg_len: if True (and targets are specified), sort source- and target batches by target length

    Returns:
      a tuple of batches if targets were given, otherwise a single batch
    """
    if trg_sents is not None and sort_by_trg_len:
      src_sents, trg_sents = zip(*sorted(zip(src_sents, trg_sents), key=lambda x: x[1].sent_len(), reverse=True))
    src_batch = pad(src_sents, pad_to_multiple=self.pad_src_to_multiple)
    if trg_sents is None:
      return src_batch
    else:
      trg_batch = pad(trg_sents)
      return src_batch, trg_batch

  def _add_single_batch(self, src_curr, trg_curr, src_ret, trg_ret, sort_by_trg_len=False):
    if trg_curr:
      src_batch, trg_batch = self.create_single_batch(src_curr, trg_curr, sort_by_trg_len)
      trg_ret.append(trg_batch)
    else:
      src_batch = self.create_single_batch(src_curr, trg_curr, sort_by_trg_len)
    src_ret.append(src_batch)

  def _pack_by_order(self,
                     src: Sequence[sent.Sentence],
                     trg: Optional[Sequence[sent.Sentence]],
                     order: Sequence[int]) -> Tuple[Sequence[xnmt.Batch], Sequence[xnmt.Batch]]:
    """
    Pack batches by given order.

    Trg is optional for the case of self.granularity == 'sent'

    Args:
      src: src-side inputs
      trg: trg-side inputs
      order: order of inputs

    Returns:
      If trg is given: tuple of src / trg batches; Otherwise: only src batches
    """
    src_ret, src_curr = [], []
    trg_ret, trg_curr = [], []
    if self.granularity == 'sent':
      for x in range(0, len(order), self.batch_size):
        src_selected = [src[y] for y in order[x:x + self.batch_size]]
        if trg:
          trg_selected = [trg[y] for y in order[x:x + self.batch_size]]
        else: trg_selected = None
        self._add_single_batch(src_selected,
                               trg_selected,
                               src_ret, trg_ret,
                               sort_by_trg_len=self.sort_within_by_trg_len)
    elif self.granularity == 'word':
      max_src, max_trg = 0, 0
      for i in order:
        max_src = max(_len_or_zero(src[i]), max_src)
        max_trg = max(_len_or_zero(trg[i]), max_trg)
        if (max_src + max_trg) * (len(src_curr) + 1) > self.batch_size and len(src_curr) > 0:
          self._add_single_batch(src_curr, trg_curr, src_ret, trg_ret, sort_by_trg_len=self.sort_within_by_trg_len)
          max_src = _len_or_zero(src[i])
          max_trg = _len_or_zero(trg[i])
          src_curr = [src[i]]
          trg_curr = [trg[i]]
        else:
          src_curr.append(src[i])
          trg_curr.append(trg[i])
      self._add_single_batch(src_curr, trg_curr, src_ret, trg_ret, sort_by_trg_len=self.sort_within_by_trg_len)
    else:
      raise RuntimeError("Illegal granularity specification {}".format(self.granularity))
    if trg:
      return src_ret, trg_ret
    else:
      return src_ret

  def pack(self, src: Sequence[sent.Sentence], trg: Sequence[sent.Sentence]) \
          -> Tuple[Sequence[xnmt.Batch], Sequence[xnmt.Batch]]:
    """
    Create a list of src/trg batches based on provided src/trg inputs.

    Args:
      src: list of src-side inputs
      trg: list of trg-side inputs

    Returns:
      tuple of lists of src and trg batches
    """
    raise NotImplementedError("must be implemented by subclasses")

class InOrderBatcher(Batcher, xnmt.Serializable):
  """
  A class to create batches in order of the original corpus, both across and within batches.

  Args:
    batch_size: batch size
    pad_src_to_multiple: pad source sentences so its length is multiple of this integer.
  """
  yaml_tag = "!InOrderBatcher"

  @xnmt.serializable_init
  def __init__(self,
               batch_size: int = 1,
               pad_src_to_multiple: int = 1) -> None:
    super().__init__(batch_size, pad_src_to_multiple=pad_src_to_multiple, sort_within_by_trg_len=False)

  def pack(self, src: Sequence[sent.Sentence], trg: Optional[Sequence[sent.Sentence]]) \
          -> Tuple[Sequence[xnmt.Batch], Sequence[xnmt.Batch]]:
    """
    Pack batches. Unlike other batches, the trg sentences are optional.

    Args:
      src: list of src-side inputs
      trg: optional list of trg-side inputs

    Returns:
      src batches if trg was not given; tuple of src batches and trg batches if trg was given
    """
    order = list(range(len(src)))
    return self._pack_by_order(src, trg, order)


class ShuffleBatcher(Batcher):
  """
  A template class to create batches through randomly shuffling without sorting.

  Sentences inside each batch are sorted by reverse trg length.

  Args:
    batch_size: batch size
    granularity: 'sent' or 'word'
    pad_src_to_multiple: pad source sentences so its length is multiple of this integer.
  """

  def __init__(self,
               batch_size: int,
               granularity: str = 'sent',
               pad_src_to_multiple: int = 1) -> None:
    super().__init__(batch_size=batch_size, granularity=granularity, pad_src_to_multiple=pad_src_to_multiple,
                     sort_within_by_trg_len=True)

  def pack(self, src: Sequence[sent.Sentence], trg: Optional[Sequence[sent.Sentence]]) -> \
      Tuple[Sequence[xnmt.Batch], Sequence[xnmt.Batch]]:
    order = list(range(len(src)))
    np.random.shuffle(order)
    return self._pack_by_order(src, trg, order)

  def is_random(self) -> bool:
    return True


class SortBatcher(Batcher):
  """
  A template class to create batches through bucketing sentence length.

  Sentences inside each batch are sorted by reverse trg length.

  Args:
    batch_size: batch size
    granularity: 'sent' or 'word'
    pad_src_to_multiple: pad source sentences so its length is multiple of this integer.
  """
  __tiebreaker_eps = 1.0e-7

  def __init__(self,
               batch_size: int,
               granularity: str = 'sent',
               sort_key: Callable = lambda x: x[0].sent_len(),
               break_ties_randomly: bool=True,
               pad_src_to_multiple: int = 1) -> None:
    super().__init__(batch_size, granularity=granularity,
                     pad_src_to_multiple=pad_src_to_multiple,
                     sort_within_by_trg_len=True)
    self.sort_key = sort_key
    self.break_ties_randomly = break_ties_randomly

  def pack(self, src: Sequence[sent.Sentence], trg: Optional[Sequence[sent.Sentence]]) \
          -> Tuple[Sequence[xnmt.Batch], Sequence[xnmt.Batch]]:
    if self.break_ties_randomly:
      order = np.argsort([self.sort_key(x) + random.uniform(-SortBatcher.__tiebreaker_eps, SortBatcher.__tiebreaker_eps) for x in zip(src,trg)])
    else:
      order = np.argsort([self.sort_key(x) for x in zip(src,trg)])
    return self._pack_by_order(src, trg, order)

  def is_random(self) -> bool:
    return self.break_ties_randomly

class SrcBatcher(SortBatcher, xnmt.Serializable):
  """
  A batcher that creates fixed-size batches, grouped by src len.

  Sentences inside each batch are sorted by reverse trg length.

  Args:
    batch_size: batch size
    break_ties_randomly: if True, randomly shuffle sentences of the same src length before creating batches.
    pad_src_to_multiple: pad source sentences so its length is multiple of this integer.
  """
  yaml_tag = "!SrcBatcher"

  @xnmt.serializable_init
  def __init__(self,
               batch_size: int,
               break_ties_randomly: bool = True,
               pad_src_to_multiple: int = 1) -> None:
    super().__init__(batch_size, sort_key=lambda x: x[0].sent_len(), granularity='sent',
                     break_ties_randomly=break_ties_randomly, pad_src_to_multiple=pad_src_to_multiple)


class TrgBatcher(SortBatcher, xnmt.Serializable):
  """
  A batcher that creates fixed-size batches, grouped by trg len.

  Sentences inside each batch are sorted by reverse trg length.

  Args:
    batch_size: batch size
    break_ties_randomly: if True, randomly shuffle sentences of the same src length before creating batches.
    pad_src_to_multiple: pad source sentences so its length is multiple of this integer.
  """
  yaml_tag = "!TrgBatcher"

  @xnmt.serializable_init
  def __init__(self,
               batch_size: int,
               break_ties_randomly: bool = True,
               pad_src_to_multiple: int = 1) -> None:
    super().__init__(batch_size, sort_key=lambda x: x[1].sent_len(), granularity='sent',
                     break_ties_randomly=break_ties_randomly,
                     pad_src_to_multiple=pad_src_to_multiple)


class SrcTrgBatcher(SortBatcher, xnmt.Serializable):
  """
  A batcher that creates fixed-size batches, grouped by src len, then trg len.

  Sentences inside each batch are sorted by reverse trg length.

  Args:
    batch_size: batch size
    break_ties_randomly: if True, randomly shuffle sentences of the same src length before creating batches.
    pad_src_to_multiple: pad source sentences so its length is multiple of this integer.
  """
  yaml_tag = "!SrcTrgBatcher"

  @xnmt.serializable_init
  def __init__(self,
               batch_size: int,
               break_ties_randomly: bool = True,
               pad_src_to_multiple: int = 1) -> None:
    super().__init__(batch_size, sort_key=lambda x: x[0].sent_len() + 1.0e-6 * len(x[1]),
                     granularity='sent', break_ties_randomly=break_ties_randomly,
                     pad_src_to_multiple=pad_src_to_multiple)


class TrgSrcBatcher(SortBatcher, xnmt.Serializable):
  """
  A batcher that creates fixed-size batches, grouped by trg len, then src len.

  Sentences inside each batch are sorted by reverse trg length.

  Args:
    batch_size: batch size
    break_ties_randomly: if True, randomly shuffle sentences of the same src length before creating batches.
    pad_src_to_multiple: pad source sentences so its length is multiple of this integer.
  """
  yaml_tag = "!TrgSrcBatcher"

  @xnmt.serializable_init
  def __init__(self,
               batch_size: int,
               break_ties_randomly: bool = True,
               pad_src_to_multiple: int = 1) -> None:
    super().__init__(batch_size, sort_key=lambda x: x[1].sent_len() + 1.0e-6 * len(x[0]),
                     granularity='sent',
                     break_ties_randomly=break_ties_randomly,
                     pad_src_to_multiple=pad_src_to_multiple)


class SentShuffleBatcher(ShuffleBatcher, xnmt.Serializable):
  """

  A batcher that creates fixed-size batches of random order.

  Sentences inside each batch are sorted by reverse trg length.

  Args:
    batch_size: batch size
    pad_src_to_multiple: pad source sentences so its length is multiple of this integer.
  """
  yaml_tag = "!SentShuffleBatcher"

  @xnmt.serializable_init
  def __init__(self, batch_size: int, pad_src_to_multiple: int = 1) -> None:
    super().__init__(batch_size, granularity='sent', pad_src_to_multiple=pad_src_to_multiple)


class WordShuffleBatcher(ShuffleBatcher, xnmt.Serializable):
  """
  A batcher that creates fixed-size batches, grouped by src len.

  Sentences inside each batch are sorted by reverse trg length.

  Args:
    words_per_batch: number of src+trg words in each batch
    pad_src_to_multiple: pad source sentences so its length is multiple of this integer.
  """
  yaml_tag = "!WordShuffleBatcher"

  @xnmt.serializable_init
  def __init__(self, words_per_batch: int, pad_src_to_multiple: int = 1) -> None:
    super().__init__(words_per_batch, granularity='word', pad_src_to_multiple=pad_src_to_multiple)


class WordSortBatcher(SortBatcher):
  """
  Base class for word sort-based batchers.

  Sentences inside each batch are sorted by reverse trg length.

  Args:
    words_per_batch: number of src+trg words in each batch
    avg_batch_size: avg number of sentences in each batch (if words_per_batch not given)
    sort_key:
    break_ties_randomly: if True, randomly shuffle sentences of the same src length before creating batches.
    pad_src_to_multiple: pad source sentences so its length is multiple of this integer.
  """

  def __init__(self,
               words_per_batch: Optional[int],
               avg_batch_size: Optional[numbers.Real],
               sort_key: Callable,
               break_ties_randomly: bool = True,
               pad_src_to_multiple: int = 1) -> None:
    # Sanity checks
    if words_per_batch and avg_batch_size:
      raise ValueError("words_per_batch and avg_batch_size are mutually exclusive.")
    elif words_per_batch is None and avg_batch_size is None:
      raise ValueError("either words_per_batch or avg_batch_size must be specified.")

    super().__init__(words_per_batch, sort_key=sort_key, granularity='word',
                     break_ties_randomly=break_ties_randomly,
                     pad_src_to_multiple=pad_src_to_multiple)
    self.avg_batch_size = avg_batch_size


class WordSrcBatcher(WordSortBatcher, xnmt.Serializable):
  """
  A batcher that creates variable-sized batches with given average (src+trg) words per batch, grouped by src len.

  Sentences inside each batch are sorted by reverse trg length.

  Args:
    words_per_batch: number of src+trg words in each batch
    avg_batch_size: avg number of sentences in each batch (if words_per_batch not given)
    break_ties_randomly: if True, randomly shuffle sentences of the same src length before creating batches.
    pad_src_to_multiple: pad source sentences so its length is multiple of this integer.
  """
  yaml_tag = "!WordSrcBatcher"

  @xnmt.serializable_init
  def __init__(self,
               words_per_batch: Optional[int] = None,
               avg_batch_size: Optional[numbers.Real] = None,
               break_ties_randomly: bool = True,
               pad_src_to_multiple: int = 1) -> None:
    super().__init__(words_per_batch, avg_batch_size, sort_key=lambda x: x[0].sent_len(),
                     break_ties_randomly=break_ties_randomly,
                     pad_src_to_multiple=pad_src_to_multiple)

  def _pack_by_order(self, src, trg, order):
    if self.avg_batch_size:
      self.batch_size = (sum([s.sent_len() for s in src]) + sum([s.sent_len() for s in trg])) / len(src) * self.avg_batch_size
    return super()._pack_by_order(src, trg, order)


class WordTrgBatcher(WordSortBatcher, xnmt.Serializable):
  """
  A batcher that creates variable-sized batches with given average (src+trg) words per batch, grouped by trg len.

  Sentences inside each batch are sorted by reverse trg length.

  Args:
    words_per_batch: number of src+trg words in each batch
    avg_batch_size: avg number of sentences in each batch (if words_per_batch not given)
    break_ties_randomly: if True, randomly shuffle sentences of the same src length before creating batches.
    pad_src_to_multiple: pad source sentences so its length is multiple of this integer.
  """
  yaml_tag = "!WordTrgBatcher"

  @xnmt.serializable_init
  def __init__(self,
               words_per_batch: Optional[int] = None,
               avg_batch_size: Optional[numbers.Real] = None,
               break_ties_randomly: bool = True,
               pad_src_to_multiple: int = 1) -> None:
    super().__init__(words_per_batch, avg_batch_size, sort_key=lambda x: x[1].sent_len(),
                     break_ties_randomly=break_ties_randomly,
                     pad_src_to_multiple=pad_src_to_multiple)

  def _pack_by_order(self, src, trg, order):
    if self.avg_batch_size:
      self.batch_size = (sum([s.sent_len() for s in src]) + sum([s.sent_len() for s in trg])) / len(src) * self.avg_batch_size
    return super()._pack_by_order(src, trg, order)


class WordSrcTrgBatcher(WordSortBatcher, xnmt.Serializable):
  """
  A batcher that creates variable-sized batches with given average number of src + trg words per batch, grouped by src len, then trg len.

  Sentences inside each batch are sorted by reverse trg length.

  Args:
    words_per_batch: number of src+trg words in each batch
    avg_batch_size: avg number of sentences in each batch (if words_per_batch not given)
    break_ties_randomly: if True, randomly shuffle sentences of the same src length before creating batches.
    pad_src_to_multiple: pad source sentences so its length is multiple of this integer.
  """
  yaml_tag = "!WordSrcTrgBatcher"

  @xnmt.serializable_init
  def __init__(self,
               words_per_batch: Optional[int] = None,
               avg_batch_size: Optional[numbers.Real] = None,
               break_ties_randomly: bool = True,
               pad_src_to_multiple: int = 1) -> None:
    super().__init__(words_per_batch, avg_batch_size, sort_key=lambda x: x[0].sent_len() + 1.0e-6 * x[1].sent_len(),
                     break_ties_randomly=break_ties_randomly,
                     pad_src_to_multiple=pad_src_to_multiple)

  def _pack_by_order(self, src, trg, order):
    if self.avg_batch_size:
      self.batch_size = (sum([s.sent_len() for s in src]) + sum([s.sent_len() for s in trg])) / len(src) * self.avg_batch_size
    return super()._pack_by_order(src, trg, order)


class WordTrgSrcBatcher(WordSortBatcher, xnmt.xnmt.Serializable):
  """
  A batcher that creates variable-sized batches with given average number of src + trg words per batch, grouped by trg len, then src len.

  Sentences inside each batch are sorted by reverse trg length.

  Args:
    words_per_batch: number of src+trg words in each batch
    avg_batch_size: avg number of sentences in each batch (if words_per_batch not given)
    break_ties_randomly: if True, randomly shuffle sentences of the same src length before creating batches.
    pad_src_to_multiple: pad source sentences so its length is multiple of this integer.
  """
  yaml_tag = "!WordTrgSrcBatcher"

  @xnmt.serializable_init
  def __init__(self,
               words_per_batch: Optional[int] = None,
               avg_batch_size: Optional[numbers.Real] = None,
               break_ties_randomly: bool = True,
               pad_src_to_multiple: int = 1) -> None:
    super().__init__(words_per_batch, avg_batch_size, sort_key=lambda x: x[1].sent_len() + 1.0e-6 * x[0].sent_len(),
                     break_ties_randomly=break_ties_randomly,
                     pad_src_to_multiple=pad_src_to_multiple)

  def _pack_by_order(self, src, trg, order):
    if self.avg_batch_size:
      self.batch_size = (sum([s.sent_len() for s in src]) + sum([s.sent_len() for s in trg])) / len(src) * self.avg_batch_size
    return super()._pack_by_order(src, trg, order)


######################################
#### Module level functions
######################################

def pad(batch_sent: collections.Sequence, pad_to_multiple: int = 1) -> xnmt.Batch:
  """
  Apply padding to sentences in a batch.

  Args:

  Returns:
    batch containing padded items and a corresponding batch mask.
  """

  if isinstance(list(batch_sent)[0], sent.CompoundSentence):
    ret = []
    for compound_i in range(len(batch_sent[0].sents)):
      ret.append(
        pad(tuple(inp.sents[compound_i] for inp in batch_sent), pad_to_multiple=pad_to_multiple))
    return batch.CompoundBatch(*ret)
  max_len = max(_len_or_zero(item) for item in batch_sent)
  if max_len % pad_to_multiple != 0:
    max_len += pad_to_multiple - (max_len % pad_to_multiple)
  min_len = min(_len_or_zero(item) for item in batch_sent)
  if min_len == max_len:
    return batch.ListBatch(batch_sent, mask=None)
  masks = np.zeros([len(batch_sent), max_len])
  for i, v in enumerate(batch_sent):
    for j in range(_len_or_zero(v), max_len):
      masks[i,j] = 1.0
  padded_items = [item.create_padded_sent(max_len - item.sent_len()) for item in batch_sent]
  return batch.ListBatch(padded_items, mask=xnmt.Mask(masks))


def pad_embedding(embeddings) -> xnmt.ExpressionSequence:
  max_col = max(len(xs) for xs in embeddings)
  p0 = dy.zeros(embeddings[0][0].dim()[0][0])
  masks = np.zeros((len(embeddings), max_col), dtype=int)
  modified = False
  ret = []
  for xs, mask in zip(embeddings, masks):
    deficit = max_col - len(xs)
    if deficit > 0:
      xs = xs + ([p0] * deficit)
      mask[-deficit:] = 1
      modified = True
    ret.append(dy.concatenate_cols(xs))
  mask = xnmt.Mask(masks) if modified else None
  return xnmt.ExpressionSequence(expr_tensor=dy.concatenate_to_batch(ret), mask=mask)


def _len_or_zero(val):
  return val.sent_len() if hasattr(val, 'sent_len') else len(val) if hasattr(val, '__len__') else 0

