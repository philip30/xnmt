
import dynet as dy
import xnmt

from typing import List, Dict, Optional

class AttenderState(object):
  def __init__(self,
               curr_sent: xnmt.ExpressionSequence,
               sent_context: dy.Expression,
               attention: Optional[dy.Expression] = None):
    self.curr_sent = curr_sent
    self.sent_context = sent_context
    self.attention = attention


class UniDirectionalState(object):
  def add_input(self, word, mask: xnmt.Mask=None) -> 'UniDirectionalState':
    raise NotImplementedError()

  def output(self) -> dy.Expression:
    pass

class DecoderState(object):
  """A state that holds whatever information is required for the decoder.
     Child classes must implement the as_vector() method, which will be
     used by e.g. the attention mechanism"""
  def as_vector(self) -> dy.Expression:
    raise  NotImplementedError()


class SentenceStats(object):
  """
  to Populate the src and trg sents statistics.
  """

  def __init__(self) -> None:
    self.src_stat = {}
    self.trg_stat = {}
    self.max_pairs = 1000000
    self.num_pair = 0

  class SourceLengthStat:
    def __init__(self) -> None:
      self.num_sents = 0
      self.trg_len_distribution = {}

  class TargetLengthStat:
    def __init__(self) -> None:
      self.num_sents = 0

  def add_sent_pair_length(self, src_length, trg_length):
    src_len_stat = self.src_stat.get(src_length, self.SourceLengthStat())
    src_len_stat.num_sents += 1
    src_len_stat.trg_len_distribution[trg_length] = \
      src_len_stat.trg_len_distribution.get(trg_length, 0) + 1
    self.src_stat[src_length] = src_len_stat

    trg_len_stat = self.trg_stat.get(trg_length, self.TargetLengthStat())
    trg_len_stat.num_sents += 1
    self.trg_stat[trg_length] = trg_len_stat

  def populate_statistics(self, train_corpus_src, train_corpus_trg):
    self.num_pair = min(len(train_corpus_src), self.max_pairs)
    for sent_num, (src, trg) in enumerate(zip(train_corpus_src, train_corpus_trg)):
      self.add_sent_pair_length(len(src), len(trg))
      if sent_num > self.max_pairs:
        return


class FinalTransducerState(object):
  """
  Represents the final encoder state; Currently handles a main (hidden) state and a cell
  state. If cell state is not provided, it is created as tanh^{-1}(hidden state).
  Could in the future be extended to handle dimensions other than h and c.

  Args:
    main_expr: expression for hidden state
    cell_expr: expression for cell state, if exists
  """
  def __init__(self, main_expr: dy.Expression, cell_expr: dy.Expression=None) -> None:
    self._main_expr = main_expr
    self._cell_expr = cell_expr

  def main_expr(self) -> dy.Expression:
    return self._main_expr

  def cell_expr(self) -> dy.Expression:
    """Returns:
         dy.Expression: cell state; if not given, it is inferred as inverse tanh of main expression
    """
    if self._cell_expr is None:
      # TODO: This taking of the tanh inverse is disabled, because it can cause NaNs
      #       Instead just copy
      # self._cell_expr = 0.5 * dy.log( dy.cdiv(1.+self._main_expr, 1.-self._main_expr) )
      self._cell_expr = self._main_expr
    return self._cell_expr


class EncoderState(object):
  def __init__(self, encode_seq: xnmt.ExpressionSequence, encoder_final_states: List[FinalTransducerState]):
    self.encode_seq = encode_seq
    self.encoder_final_states = encoder_final_states


class SearchAction(object):
  def __init__(self,
               action: int,
               log_likelihood: dy.Expression=None,
               decoder_state: DecoderState=None,
               mask: xnmt.Mask=None):
    self._action = action
    self._log_likelihood = log_likelihood
    self._mask = mask
    self._decoder_state = decoder_state

  @property
  def action(self):
    return self._action

  @property
  def log_likelihood(self):
    return self._log_likelihood

  @property
  def mask(self):
    return self._mask

  @property
  def decoder_state(self):
    return self._decoder_state

  def __repr__(self):
    ll = dy.exp(self.log_likelihood).npvalue() if self.log_likelihood is not None else None
    return "({}, {})".format(repr(self.action), ll)


