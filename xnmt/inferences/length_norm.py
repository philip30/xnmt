import numpy as np

from scipy.stats import norm
from typing import Sequence, Optional

import xnmt
import xnmt.models as models


class NoNormalization(models.LengthNormalization, xnmt.Serializable):
  """
  Adding no form of length normalization.
  """
  @xnmt.serializable_init
  def __init__(self):
    pass

  def normalize_completed(self, completed_hyps: Sequence[models.Hypothesis],
                          src_length: Optional[int] = None) -> Sequence[float]:
    return [hyp.score for hyp in completed_hyps]


class AdditiveNormalization(models.LengthNormalization, xnmt.Serializable):
  """
  Adding a fixed word penalty everytime the word is added.
  """
  @xnmt.serializable_init
  def __init__(self, penalty: float = -0.1, apply_during_search: bool = False):
    self.penalty = penalty
    self.apply_during_search = apply_during_search

  def normalize_completed(self, completed_hyps: Sequence[models.Hypothesis],
                          src_length: Optional[int] = None) -> Sequence[float]:
    if self.apply_during_search:
      return [hyp.score for hyp in completed_hyps]
    else:
      return [hyp.score + (hyp.timestep * self.penalty) for hyp in completed_hyps]

  def normalize_partial_topk(self, score_so_far, score_to_add, new_len):
    return score_so_far + score_to_add + (self.penalty if self.apply_during_search else 0.0)


class PolynomialNormalization(models.LengthNormalization, xnmt.Serializable):
  """
  Dividing by the length (raised to some power)
  """
  @xnmt.serializable_init
  def __init__(self, m: float = 1, apply_during_search: bool = False):
    self.m = m
    self.apply_during_search = apply_during_search
    self.pows = []

  def normalize_completed(self, completed_hyps: Sequence[models.Hypothesis],
                          src_length: Optional[int] = None) -> Sequence[float]:
    if self.apply_during_search:
      return [hyp.score for hyp in completed_hyps]
    else:
      return [(hyp.score / pow(hyp.timestep, self.m)) for hyp in completed_hyps]
    
  def normalize_partial_topk(self, score_so_far, score_to_add, new_len):
    if self.apply_during_search:
      self.update_pows(new_len)
      return (score_so_far * self.pows[new_len-1] + score_to_add) / self.pows[new_len]
    else:
      return score_so_far + score_to_add
 
  def update_pows(self, new_len):
    if len(self.pows) < new_len+1:
      for i in range(len(self.pows), new_len+1):
        self.pows.append(pow(i, self.m))


class MultinomialNormalization(models.LengthNormalization, xnmt.Serializable):
  """
  The algorithm followed by:
  Tree-to-Sequence Attentional Neural Machine Translation
  https://arxiv.org/pdf/1603.06075.pdf
  """
  @xnmt.serializable_init
  def __init__(self, sent_stats):
    self.stats = sent_stats

  def trg_length_prob(self, src_length, trg_length):
    v = len(self.stats.src_stat)
    if src_length in self.stats.src_stat:
      src_stat = self.stats.src_stat.get(src_length)
      return (src_stat.trg_len_distribution.get(trg_length, 0) + 1) / (src_stat.num_sents + v)
    return 1

  def normalize_completed(self, completed_hyps: Sequence[models.Hypothesis],
                          src_length: Optional[int] = None) -> Sequence[float]:
    """
    Args:
      completed_hyps:
      src_length: length of the src sent
    """
    assert (src_length is not None), "Length of Source Sentence is required"

    return [hyp.score + np.log(self.trg_length_prob(src_length, hyp.timestep)) for hyp in completed_hyps]


class GaussianNormalization(models.LengthNormalization, xnmt.Serializable):
  """
   The Gaussian regularization encourages the inference
   to select sents that have similar lengths as the
   sents in the training set.
   refer: https://arxiv.org/pdf/1509.04942.pdf
  """
  @xnmt.serializable_init
  def __init__(self, sent_stats: models.SentenceStats):
    self.stats = sent_stats.trg_stat
    self.num_sent = sent_stats.num_pair
    y = np.zeros(self.num_sent)
    curr_iter = 0
    for key in self.stats:
      iter_end = self.stats[key].num_sents + curr_iter
      y[curr_iter:iter_end] = key
      curr_iter = iter_end
    mu, std = norm.fit(y)
    self.distr = norm(mu, std)

  def trg_length_prob(self, trg_length):
    return self.distr.pdf(trg_length)

  def normalize_completed(self, completed_hyps: Sequence[models.Hypothesis],
                          src_length: Optional[int] = None) -> Sequence[float]:
    return [hyp.score / self.trg_length_prob(hyp.timestep) for hyp in completed_hyps]


class EosBooster(xnmt.Serializable):
  """
  Callable that applies boosting of end-of-sequence token, can be used with :class:`xnmt.search_strategy.BeamSearch`.

  Args:
    boost_val: value to add to the eos token's log probability. Positive values make sentences shorter, negative values
               make sentences longer.
  """
  @xnmt.serializable_init
  def __init__(self, boost_val: float):
    self.boost_val = boost_val
  
  def __call__(self, scores:np.ndarray):
    scores[xnmt.Vocab.ES] += self.boost_val
