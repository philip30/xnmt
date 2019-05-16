from typing import List, Union

import xnmt
import xnmt.models as models
import xnmt.inferences.length_norm as norms


class GreedySearch(models.SearchStrategy, xnmt.Serializable):
  """
  Performs greedy search (aka beam search with beam size 1)

  Args:
    max_len: maximum number of tokens to generate.
  """
  @xnmt.serializable_init
  def __init__(self, max_len: int = 100):
    self.max_len = max_len

  def generate_output(self,
                      generator: Union[models.GeneratorModel, models.AutoRegressiveModel],
                      initial_state: models.DecoderState) -> List[models.Hypothesis]:
    # Search Variables
    hyp = models.Hypothesis(0, models.SearchAction(initial_state))
    for length in range(self.max_len):
      prev_word = hyp.action.action_id
      
      if generator.finish_generating(prev_word, hyp.action.decoder_state):
        break
      
      next_state = generator.add_input(prev_word, hyp.action.decoder_state)
      next_action = generator.best_k(next_state, 1, normalize_scores=True)[0]
      next_score = hyp.score + next_action.log_likelihood
      hyp = models.Hypothesis(score=next_score, action=next_action, timestep=hyp.timestep+1, parent=hyp)
    return [hyp]


class BeamSearch(models.SearchStrategy, xnmt.Serializable):
  """
  Performs beam search.

  Args:
    beam_size: number of beams
    max_len: maximum number of tokens to generate.
    len_norm: type of length normalization to apply
  """
  @xnmt.serializable_init
  def __init__(self,
               beam_size: int = 1,
               top_k: int = 1,
               max_len: int = 100,
               len_norm: models.LengthNormalization = xnmt.bare(norms.NoNormalization)):
    self.beam_size = beam_size
    self.max_len = max_len
    self.len_norm = len_norm
    self.top_k = top_k

  def generate_output(self,
                      generator: Union[models.AutoRegressiveModel, models.GeneratorModel],
                      initial_state: models.DecoderState) -> List[models.Hypothesis]:
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
        # Find the k best words at the next time step
        next_state = generator.add_input(xnmt.mark_as_batch(prev_word), prev_state)
        next_actions = generator.best_k(next_state, self.beam_size, normalize_scores=True)
        # Queue next states
        for action in next_actions:
          new_score = self.len_norm.normalize_partial_topk(hyp.score, action.log_likelihood, length+1)
          new_set.append(models.Hypothesis(new_score, action, hyp.timestep+1, hyp))
      # Next top hypothesis
      active_hyp = sorted(new_set, key=lambda x: x.score, reverse=True)[:self.beam_size]
    # There is no hyp that reached </s>
    if len(completed_hyp) == 0:
      completed_hyp = active_hyp

    # Length Normalization
    src_length = xnmt.globals.singleton_global.src_batch[0].len_unpadded()
    normalized_scores = self.len_norm.normalize_completed(completed_hyp, src_length)
    normalized_hyp = []
    for hyp, score in zip(completed_hyp, normalized_scores):
      normalized_hyp.append(models.Hypothesis(score, hyp.action, hyp.timestep, hyp.parent))
    normalized_hyp = sorted(normalized_hyp, key=lambda x: x.score, reverse=True)
    
    return normalized_hyp


class SamplingSearch(models.SearchStrategy, xnmt.Serializable):
  """
  Performs search based on the softmax probability distribution.
  Similar to greedy searchol

  Args:
    max_len:
    sample_size:
  """
  @xnmt.serializable_init
  def __init__(self, max_len: int = 100, sample_size: int = 5):
    self.max_len = max_len
    self.sample_size = sample_size

  def generate_output(self,
                      generator: Union[models.GeneratorModel, models.AutoRegressiveModel],
                      initial_state: models.DecoderState) -> List[models.Hypothesis]:
    hyp = models.Hypothesis(0, models.SearchAction(initial_state))
    hyps = [hyp] * self.sample_size
    done_flag = [False] * self.sample_size
   
    # Sample to the max length
    for length in range(self.max_len):
      new_hyps = []
      for i in range(self.sample_size):
        prev_word = hyps[i].action.action_id
        hyp = hyps[i]
        if done_flag[i] or generator.finish_generating(prev_word, hyp.action.decoder_state):
          done_flag[i] = True
          new_hyps.append(hyp)
        else:
          next_state = generator.add_input(xnmt.mark_as_batch(prev_word), hyp.action.decoder_state)
          next_action = generator.sample(next_state, 1)[0]
          next_score = hyp.score + next_action.log_likelihood
          hyp = models.Hypothesis(score=next_score, action=next_action, timestep=hyp.timestep+1, parent=hyp)
          new_hyps.append(hyp)
      hyps = new_hyps
      
      if all(done_flag):
        break
      
    return hyps


