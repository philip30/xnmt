from typing import List, Union, Optional

import xnmt
import xnmt.models as models
import xnmt.inferences.length_norm as norms


class GreedySearch(models.SearchStrategy, xnmt.Serializable, models.ForceableSearchStrategy):
  yaml_tag = "!GreedySearch"
  """
  Performs greedy search (aka beam search with beam size 1)

  Args:
    max_len: maximum number of tokens to generate.
  """
  @xnmt.serializable_init
  def __init__(self, max_len: int = 100, is_forced=False, is_sampling=False):
    xnmt.Serializable.__init__(self)
    self._max_len = max_len
    self._is_forced = is_forced
    self._is_sampling = is_sampling

  def generate_forced_output(self,
                      generator: models.AutoRegressiveModel,
                      src: Optional[xnmt.Batch],
                      trg: Optional[xnmt.Batch]) -> List[models.Hypothesis]:
    initial_state = generator.initial_state(src)
    not_forced = trg is None
    seq_len = self._max_len if not_forced else trg.sent_len()
    
    # Search Variables
    hyp = models.Hypothesis(0, models.SearchAction(initial_state))
    for length in range(seq_len):
      prev_word = hyp.action.action_id
      
      ## If not forced (normal generation), end of the generation depends on the
      # seq_len or decided by the generator
      if not_forced:
        if generator.finish_generating(prev_word, hyp.action.decoder_state):
          break
          
      ## Case first item in the sequence, input the network with special BOS item
      # to mark the beginning of a generation
      if prev_word is None:
        prev_word = xnmt.mark_as_batch([xnmt.Vocab.SS] * src.batch_size())
      
      ## Feeding the prev input to the generator to generate the next state
      next_state = generator.add_input(prev_word, hyp.action.decoder_state)
      
      ## Decide the next action
      if not_forced:
        if self._is_sampling:
          next_action = generator.sample(next_state, 1)[0]
        else:
          next_action = generator.best_k(next_state, 1, normalize_scores=True)[0]
      else:
        ref_word = xnmt.mark_as_batch([trg[i][length] for i in range(trg.batch_size())])
        next_action = generator.pick_oracle(next_state, ref_word)[0]
     
      ## Hypothesis of t+1 (next hyp)
      next_score = hyp.score + next_action.log_likelihood.value()
      hyp = models.Hypothesis(score=next_score, action=next_action, timestep=hyp.timestep+1, parent=hyp)
    return [hyp]

  def generate_output(self,
                      generator: Union[models.GeneratorModel, models.AutoRegressiveModel],
                      src: xnmt.Batch):
    return self.generate_forced_output(generator, src, None)

  def is_forced(self):
    return self._is_forced


class BeamSearch(models.SearchStrategy, xnmt.Serializable):
  yaml_tag = "!BeamSearch"
  """
  Performs beam search.

  Args:
    beam_size: number of beams
    max_len: maximum number of tokens to generate.
    len_norm: type of length normalization to apply
  """
  @xnmt.serializable_init
  def __init__(self,
               beam_size: int = 5,
               top_k: int = 1,
               max_len: int = 100,
               is_sampling: bool = False,
               len_norm: models.LengthNormalization = xnmt.bare(norms.NoNormalization)):
    xnmt.Serializable.__init__(self)
    self.beam_size = beam_size
    self.max_len = max_len
    self.len_norm = len_norm
    self.top_k = top_k
    self.is_sampling = is_sampling

  def generate_output(self,
                      generator: models.AutoRegressiveModel,
                      src: xnmt.Batch) -> List[models.Hypothesis]:
    initial_state = generator.initial_state(src)
    active_hyp = [models.Hypothesis(0, models.SearchAction(initial_state))]
    completed_hyp = []
    for length in range(self.max_len):
      if len(completed_hyp) >= self.beam_size:
        break
      # Expand hyp
      new_set = []
      for hyp in active_hyp:
        prev_word = hyp.action.action_id
        prev_state = hyp.action.decoder_state
        # We have a complete hyp ending with </s>
        if generator.finish_generating(prev_word, prev_state):
          completed_hyp.append(hyp)
          continue
        if prev_word is None:
          prev_word = xnmt.mark_as_batch([xnmt.Vocab.SS] * src.batch_size())
        # Find the k best words at the next time step
        next_state = generator.add_input(prev_word, prev_state)
        if self.is_sampling:
          next_actions = generator.sample(next_state, self.beam_size)
        else:
          next_actions = generator.best_k(next_state, self.beam_size, normalize_scores=True)
        # Queue next states
        for action in next_actions:
          new_score = self.len_norm.normalize_partial_topk(hyp.score, action.log_likelihood.value(), length+1)
          new_set.append(models.Hypothesis(new_score, action, hyp.timestep+1, hyp))
      # Next top hypothesis
      active_hyp = sorted(new_set, key=lambda x: x.score, reverse=True)[:self.beam_size]
    # There is no hyp that reached </s>
    if len(completed_hyp) == 0:
      completed_hyp = active_hyp

    # Length Normalization
    src_length = initial_state.src[0].len_unpadded()
    normalized_scores = self.len_norm.normalize_completed(completed_hyp, src_length)
    normalized_hyp = []
    for hyp, score in zip(completed_hyp, normalized_scores):
      normalized_hyp.append(models.Hypothesis(score, hyp.action, hyp.timestep, hyp.parent))
    normalized_hyp = sorted(normalized_hyp, key=lambda x: x.score, reverse=True)
    normalized_hyp = normalized_hyp[:self.top_k]
    return normalized_hyp



