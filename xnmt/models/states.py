import random
import dynet as dy
import xnmt
import functools

from typing import List, Optional, Tuple


class AttenderState(object):
  def __init__(self,
               curr_sent: dy.Expression,
               sent_context: dy.Expression,
               input_mask: Optional[xnmt.Mask] = None,
               attention: Optional[dy.Expression] = None,
               initial_context: Optional[Tuple[dy.Expression, dy.Expression]] = None):
    self.curr_sent = curr_sent
    self.sent_context = sent_context
    self.initial_context = initial_context or (curr_sent, sent_context)
    self.attention = attention
    self.input_mask = input_mask


class UniDirectionalState(object):
  def add_input(self, word: dy.Expression, mask: Optional[xnmt.Mask] = None) -> 'UniDirectionalState':
    raise NotImplementedError()

  def output(self) -> dy.Expression:
    raise NotImplementedError()
  
  def context(self) -> dy.Expression:
    return self.output()


class IdentityUniDirectionalState(UniDirectionalState):
  def __init__(self, content: Optional[dy.Expression] = None):
    self.content = content
    
  def output(self):
    return self.content

  def add_input(self, word: dy.Expression, mask: Optional[xnmt.Mask] = None):
    return IdentityUniDirectionalState(word)


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
  def __init__(self, encode_seq: xnmt.ExpressionSequence, encoder_final_states: Optional[List[FinalTransducerState]]):
    self.encode_seq = encode_seq
    self.encoder_final_states = encoder_final_states


class SearchAction(object):
  def __init__(self,
               decoder_state: Optional[UniDirectionalState] = None,
               action_id: Optional[int] = None,
               log_likelihood: Optional[dy.Expression] = None,
               log_softmax: Optional[dy.Expression] = None,
               mask: Optional[xnmt.Mask] = None):
    self._action_id = action_id
    self._log_likelihood = log_likelihood
    self._log_softmax = log_softmax
    self._mask = mask
    self._decoder_state = decoder_state

  @property
  def action_id(self):
    return self._action_id

  @property
  def log_likelihood(self):
    return self._log_likelihood

  @property
  def mask(self):
    return self._mask

  @property
  def decoder_state(self):
    return self._decoder_state
  
  @property
  def log_softmax(self):
    return self._log_softmax

  def __repr__(self):
    ll = dy.exp(self.log_likelihood).npvalue() if self.log_likelihood is not None else None
    return "({}, {})".format(repr(self.action_id), ll)


class Hypothesis(object):
  def __init__(self, score: float, action: SearchAction, timestep: int = 0, parent: Optional['Hypothesis'] = None):
    self._score = score
    self._action = action
    self._timestep = timestep
    self._parent = parent

  @property
  def score(self):
    return self._score

  @property
  def action(self):
    return self._action

  @property
  def timestep(self):
    return self._timestep

  @property
  def parent(self):
    return self._parent

  @functools.lru_cache(maxsize=1)
  def actions(self):
    actions = []
    now = self
    while now.parent is not None:
      actions.append(now.action)
      now = now.parent
    return list(reversed(actions))


class TrainingState(object):
  """
  This holds the state of the training loop.
  """
  def __init__(self):
    self.num_times_lr_decayed = 0
    self.cur_attempt = 0
    self.epoch_num = 0
    self.steps_into_epoch = 0
    self.sents_since_start = 0
    self.sents_into_epoch = 0
    self.best_dev_score = None
    # used to pack and shuffle minibatches (keeping track might help resuming crashed trainings in the future)
    self.epoch_seed = random.randint(1,2147483647)
