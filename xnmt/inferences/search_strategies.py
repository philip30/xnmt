from collections import namedtuple
from typing import Callable, List, Optional
import numbers

import numpy as np

from xnmt.modules.nn.decoders import base as decoders
from xnmt.inferences.length_norm import NoNormalization, LengthNormalization
from xnmt.internal.persistence import Serializable, serializable_init, bare
from xnmt.rl import PolicyAction


"""
Output of the search
words_ids: list of generated word ids
attentions: list of corresponding attention vector of word_ids
score: a single value of log(p(E|F))
logsoftmaxes: a corresponding softmax vector of the score. score = logsoftmax[word_id]
state: a NON-BACKPROPAGATEABLE state that is used to produce the logsoftmax layer
       state is usually used to generate 'baseline' in reinforce loss
masks: whether the particular word id should be ignored or not (1 for not, 0 for yes)
"""




class GreedySearch(Serializable, SearchStrategy):
  """
  Performs greedy search (aka beam search with beam size 1)

  Args:
    max_len: maximum number of tokens to generate.
  """

  yaml_tag = '!GreedySearch'

  @serializable_init
  def __init__(self, max_len: numbers.Integral = 100) -> None:
    self.max_len = max_len

  def generate_output(self,
                      generator: 'xnmt.networks.base.GeneratorModel',
                      initial_state: decoders.DecoderState,
                      src_length: Optional[numbers.Integral] = None) -> List[PolicyAction]:
    # Output variables
    score = []
    word_ids = []
    attentions = []
    states = []
    masks = []
    # Search Variables
    done = None
    current_state = initial_state
    for length in range(self.max_len):
      prev_word = word_ids[length-1] if length > 0 else None
      current_output = generator.add_input(prev_word, current_state)
      word_id, word_score = generator.best_k(current_output.state, 1, normalize_scores=True)
      word_id = word_id[0]
      word_score = word_score[0]
      current_state = current_output.state

      if not type(word_id) == np.ndarray or len(word_id.shape) == 0:
        word_id = np.array([word_id])
        word_score = np.array([word_score])
      if done is not None:
        word_id = [word_id[i] if not done[i] else generator.eog_symbol() for i in range(len(done))]
        mask = [1 if not done[i] else 0 for i in range(len(done))]
        word_score = [s * m for (s, m) in zip(word_score, mask)]
        masks.append(mask)

      # Packing outputs
      score.append(word_score)
      word_ids.append(word_id)
      attentions.append(current_output.attention)
      states.append(current_state)

      # Check if we are done.
      done = generator.finish_generating(word_id, current_state)
      if all(done):
        break

    masks.insert(0, [1 for _ in range(len(done))])
    words = np.stack(word_ids, axis=1)
    score = np.sum(score, axis=0)
    return [SearchOutput(words, attentions, score, states, masks)]


class BeamSearch(Serializable, SearchStrategy):
  """
  Performs beam search.

  Args:
    beam_size: number of beams
    max_len: maximum number of tokens to generate.
    len_norm: type of length normalization to apply
    one_best: Whether to output the best hyp only or all completed hyps.
    scores_proc: apply an optional operation on all scores prior to choosing the top k.
                 E.g. use with :class:`xnmt.length_normalization.EosBooster`.
  """

  yaml_tag = '!BeamSearch'
  Hypothesis = namedtuple('Hypothesis', ['score', 'output', 'parent', 'word'])

  @serializable_init
  def __init__(self,
               beam_size: numbers.Integral = 1,
               max_len: numbers.Integral = 100,
               len_norm: LengthNormalization = bare(NoNormalization),
               one_best: bool = True,
               scores_proc: Optional[Callable[[np.ndarray], None]] = None) -> None:
    self.beam_size = beam_size
    self.max_len = max_len
    self.len_norm = len_norm
    self.one_best = one_best
    self.scores_proc = scores_proc

  def generate_output(self,
                      generator: 'xnmt.networks.base.GeneratorModel',
                      initial_state: decoders.DecoderState,
                      src_length: Optional[numbers.Integral] = None) -> List[PolicyAction]:
    active_hyp = [self.Hypothesis(0, None, None, None)]
    completed_hyp = []
    for length in range(self.max_len):
      if len(completed_hyp) >= self.beam_size:
        break
      # Expand hyp
      new_set = []
      for hyp in active_hyp:
        if length > 0:
          prev_word = hyp.word
          prev_state = hyp.output.state
        else:
          prev_word = None
          prev_state = initial_state
        # We have a complete hyp ending with </s>
        done = generator.finish_generating([prev_word], prev_state)
        if all(done):
          completed_hyp.append(hyp)
          continue
        # Find the k best words at the next time step
        current_output = generator.add_input(prev_word, prev_state)
        top_words, top_scores = generator.best_k(current_output.state, self.beam_size, normalize_scores=True)
        # Queue next states
        for cur_word, score in zip(top_words, top_scores):
          assert len(score.shape) == 0
          new_score = self.len_norm.normalize_partial_topk(hyp.score, score, length + 1)
          new_set.append(self.Hypothesis(new_score, current_output, hyp, cur_word))
      # Next top hypothesis
      active_hyp = sorted(new_set, key=lambda x: x.score, reverse=True)[:self.beam_size]
    # There is no hyp that reached </s>
    if len(completed_hyp) == 0:
      completed_hyp = active_hyp

    # Length Normalization
    normalized_scores = self.len_norm.normalize_completed(completed_hyp, src_length)
    hyp_and_score = sorted(list(zip(completed_hyp, normalized_scores)), key=lambda x: x[1], reverse=True)

    # Take only the one best, if that's what was desired
    if self.one_best:
      hyp_and_score = [hyp_and_score[0]]

    # Backtracing + Packing outputs
    results = []
    for end_hyp, score in hyp_and_score:
      word_ids = []
      attentions = []
      states = []
      current = end_hyp
      while current.parent is not None:
        word_ids.append(current.word)
        attentions.append(current.output.attention)
        states.append(current.output.state)
        current = current.parent
      results.append(SearchOutput([list(reversed(word_ids))], [list(reversed(attentions))],
                                  [score], list(reversed(states)), [1 for _ in word_ids]))
    return results


class SamplingSearch(Serializable, SearchStrategy):
  """
  Performs search based on the softmax probability distribution.
  Similar to greedy searchol

  Args:
    max_len:
    sample_size:
  """

  yaml_tag = '!SamplingSearch'

  @serializable_init
  def __init__(self, max_len: numbers.Integral = 100, sample_size: numbers.Integral = 5) -> None:
    self.max_len = max_len
    self.sample_size = sample_size

  def generate_output(self,
                      translator: 'xnmt.networks.base.GeneratorModel',
                      initial_state: decoders.DecoderState) -> List[PolicyAction]:
    outputs = []
    for k in range(self.sample_size):
      outputs.append(self.sample_one(translator, initial_state))
    return outputs

  # Words ids, attentions, score, logsoftmax, state
  def sample_one(self,
                 generator: 'xnmt.networks.base.GeneratorModel',
                 initial_state: decoders.DecoderState) -> PolicyAction:
    # Search variables
    current_words = None
    current_state = initial_state
    done = None
    # Outputs
    scores = []
    samples = []
    states = []
    attentions = []
    masks = []
    # Sample to the max length
    for length in range(self.max_len):
      current_output = generator.add_input(current_words, current_state)
      word_id, word_score = generator.sample(current_output.state, 1)[0]
      word_score = word_score.npvalue()
      assert word_score.shape == (1,)
      word_score = word_score[0]

      if len(word_id.shape) == 0:
        word_id = np.array([word_id])
        word_score = np.array([word_score])

      if done is not None:
        word_id = [word_id[i] if not done[i] else generator.eog_symbol() for i in range(len(done))]
        # masking for logsoftmax
        mask = [1 if not done[i] else 0 for i in range(len(done))]
        word_score = [s * m for (s, m) in zip(word_score, mask)]
        masks.append(mask)

      # Appending output
      scores.append(word_score)
      samples.append(word_id)
      states.append(current_output.state)
      attentions.append(current_output.attention)

      # Next time step
      current_words = word_id
      current_state = current_output.state

      # Check done
      done = generator.finish_generating(current_words, current_state)
      # Check if we are done.
      if all(done):
        break

    # Packing output
    scores = [np.sum(scores)]
    masks.insert(0, [1 for _ in range(len(done))])
    samples = np.stack(samples, axis=1)
    return SearchOutput(samples, attentions, scores, states, masks)




