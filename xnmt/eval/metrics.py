"""
This module contains classes to compute evaluation metrics and to hold the resulting scores.

:class:`EvalScore` subclasses represent a computed score, including useful statistics, and can be
printed with an informative string representation.

:class:`Evaluator` subclasses are used to compute these scores. Currently the following are implemented:

* :class:`LossScore` (created directly by the model)
* :class:`BLEUEvaluator` and :class:`FastBLEUEvaluator` create :class:`BLEUScore` objects
* :class:`GLEUEvaluator` creates :class:`GLEUScore` objects
* :class:`WEREvaluator` creates :class:`WERScore` objects
* :class:`CEREvaluator` creates :class:`CERScore` objects
* :class:`ExternalEvaluator` creates :class:`ExternalScore` objects
* :class:`SequenceAccuracyEvaluator` creates :class:`SequenceAccuracyScore` objects

"""


import numpy as np

from typing import Sequence, Dict, Any, Optional
import xnmt.models as models
import xnmt.modules


__all__ = ["SentenceLevelEvalScore", "LossScore", "BLEUScore", "GLEUScore",
           "LevenshteinScore", "WERScore", "CERScore", "RecallScore", "ExternalScore",
           "SequenceAccuracyScore", "FMeasureScore"]


class SentenceLevelEvalScore(models.EvalScore):
  """
  A template class for scores that work on a sentence-level and can be aggregated to corpus-level.
  """
  @staticmethod
  def aggregate(scores: Sequence['SentenceLevelEvalScore'], desc: Any = None) -> 'SentenceLevelEvalScore':
    """
    Aggregate a sequence of sentence-level scores into a corpus-level score.

    Args:
      scores: list of sentence-level scores.
      desc: human-readable description.

    Returns:
      Score object that is the aggregate of all sentence-level scores.
    """
    raise NotImplementedError()


class LossScore(models.EvalScore, xnmt.Serializable):
  yaml_tag = "!LossScore"
  """
  Score indicating the value of the loss function of a neural network.

  Args:
    loss: the (primary) loss value
    loss_stats: info on additional loss values
    num_ref_words: number of reference tokens
    desc: human-readable description to include in log outputs
  """
  @xnmt.serializable_init
  def __init__(self,
               loss: float,
               loss_stats: Dict[str, float] = None,
               num_ref_words: Optional[int] = None,
               desc: Any = None) -> None:
    super().__init__(desc=desc)
    self.loss = loss
    self.loss_stats = loss_stats
    self.num_ref_words = num_ref_words
    self.serialize_params = {"loss":loss}
    if desc is not None: self.serialize_params["desc"] = desc
    if loss_stats is not None: self.serialize_params["loss_stats"] = desc
  def value(self): return self.loss
  def metric_name(self): return "Loss"
  def higher_is_better(self): return False
  def score_str(self):
    if self.loss_stats is not None and len(self.loss_stats) > 1:
      return "{" + ", ".join(f"{k}: {v:.5f}" for k, v in self.loss_stats.items()) + f"}} (ref_len={self.num_ref_words})"
    else:
      return f"{self.value():.3f} (ref_len={self.num_ref_words})"


class BLEUScore(models.EvalScore, xnmt.Serializable):
  yaml_tag = "!BLEUScore"
  """
  Class to keep a BLEU score.

  Args:
    bleu: actual BLEU score between 0 and 1
    frac_score_list: list of fractional scores for each n-gram order
    brevity_penalty_score: brevity penalty that was multiplied to the precision score.
    hyp_len: length of hypothesis
    ref_len: length of reference
    ngram: match n-grams up to this order (usually 4)
    desc: human-readable description to include in log outputs
  """
  @xnmt.serializable_init
  def __init__(self,
               bleu: Optional[float],
               frac_score_list: Sequence[float] = None,
               brevity_penalty_score: float = None,
               hyp_len: int = None,
               ref_len: int = None,
               ngram: int = 4,
               desc: Any = None) -> None:
    self.bleu = bleu
    self.frac_score_list = frac_score_list
    self.brevity_penalty_score = brevity_penalty_score
    self.hyp_len = hyp_len
    self.ref_len = ref_len
    self.ngram   = ngram
    self.desc = desc
    self.serialize_params = {"bleu":bleu, "ngram":ngram}
    self.serialize_params.update(
      {k: getattr(self, k) for k in ["frac_score_list", "brevity_penalty_score", "hyp_len", "ref_len", "desc"] if
       getattr(self, k) is not None})

  def value(self): return self.bleu if self.bleu is not None else 0.0
  def metric_name(self): return "BLEU" + str(self.ngram)
  def higher_is_better(self): return True
  def score_str(self):
    if self.bleu is None:
      return "0"
    else:
      return f"{self.bleu}, {'/'.join(self.frac_score_list)} (BP = {self.brevity_penalty_score:.6f}, " \
             f"ratio={self.hyp_len / self.ref_len:.2f}, hyp_len={self.hyp_len}, ref_len={self.ref_len})"


class GLEUScore(SentenceLevelEvalScore, xnmt.Serializable):
  yaml_tag = "!GLEUScore"
  """
  Class to keep a GLEU (Google BLEU) score.

  Args:
    gleu: actual GLEU score between 0 and 1
    hyp_len: length of hypothesis
    ref_len: length of reference
    desc: human-readable description to include in log outputs
  """
  @xnmt.serializable_init
  def __init__(self,
               corpus_n_match: int,
               corpus_total: int,
               hyp_len: int,
               ref_len: int,
               desc: Any = None) -> None:
    self.corpus_n_match = corpus_n_match
    self.corpus_total = corpus_total
    self.hyp_len = hyp_len
    self.ref_len = ref_len
    self.desc = desc
    self.serialize_params = {"corpus_n_match": corpus_n_match, "corpus_total": corpus_total, "hyp_len":hyp_len,
                             "ref_len":ref_len}
    if desc is not None: self.serialize_params["desc"] = desc

  def value(self):
    if self.corpus_total == 0:
      return 0.0
    else:
      return self.corpus_n_match / self.corpus_total
  def metric_name(self): return "GLEU"
  def higher_is_better(self): return True
  def score_str(self):
    return "{:.6f}".format(self.value())

  @staticmethod
  def aggregate(scores: Sequence['SentenceLevelEvalScore'], desc: Any = None):
    return GLEUScore(corpus_n_match=sum(s.corpus_n_match for s in scores),
                     corpus_total=sum(s.corpus_total for s in scores),
                     hyp_len=sum(s.hyp_len for s in scores),
                     ref_len=sum(s.ref_len for s in scores),
                     desc=desc)


class LevenshteinScore(SentenceLevelEvalScore):
  """
  A template class for Levenshtein-based scores.

  Args:
    correct: number of correct matches
    substitutions: number of substitution errors
    insertions: number of insertion errors
    deletions: number of deletion errors
    desc: human-readable description to include in log outputs
  """

  @xnmt.serializable_init
  def __init__(self,
               correct: int,
               substitutions: int,
               insertions: int,
               deletions: int,
               desc: Any = None) -> None:
    self.correct = correct
    self.substitutions = substitutions
    self.insertions = insertions
    self.deletions = deletions
    self.desc = desc
    self.serialize_params = {"correct": correct, "substitutions": substitutions, "insertions": insertions,
                             "deletions": deletions}
    if desc is not None: self.serialize_params["desc"] = desc
  def value(self): return (self.substitutions + self.insertions + self.deletions) / (self.ref_len())
  def hyp_len(self):
    return self.correct + self.substitutions + self.insertions
  def ref_len(self):
    return self.correct + self.substitutions + self.deletions
  def higher_is_better(self): return False
  def score_str(self):
    return f"{self.value()*100.0:.2f}% " \
           f"( C/S/I/D: {self.correct}/{self.substitutions}/{self.insertions}/{self.deletions}; " \
           f"hyp_len={self.hyp_len()}, ref_len={self.ref_len()} )"
  @staticmethod
  def aggregate(scores: Sequence['LevenshteinScore'], desc: Any = None) -> 'LevenshteinScore':
    return scores[0].__class__(correct=sum(s.correct for s in scores),
                               substitutions=sum(s.substitutions for s in scores),
                               insertions=sum(s.insertions for s in scores),
                               deletions=sum(s.deletions for s in scores))


class WERScore(LevenshteinScore, xnmt.Serializable):
  yaml_tag = "!WERScore"
  """
  Class to keep a word error rate.
  """
  def metric_name(self): return "WER"


class CERScore(LevenshteinScore, xnmt.Serializable):
  yaml_tag = "!CERScore"
  """
  Class to keep a character error rate.
  """
  def metric_name(self): return "CER"


class RecallScore(SentenceLevelEvalScore, xnmt.Serializable):
  yaml_tag = "!RecallScore"
  """
  Class to keep a recall score.

  Args:
    recall: recall score value between 0 and 1
    hyp_len: length of hypothesis
    ref_len: length of reference
    nbest: recall computed within n-best of specified n
    desc: human-readable description to include in log outputs
  """
  @xnmt.serializable_init
  def __init__(self,
               recall: float,
               hyp_len: int,
               ref_len: int,
               nbest: int = 5,
               desc: Any = None) -> None:
    self.recall  = recall
    self.hyp_len = hyp_len
    self.ref_len = ref_len
    self.nbest   = nbest
    self.desc = desc
    self.serialize_params = {"recall":recall, "hyp_len":hyp_len,"ref_len":ref_len, "nbest":nbest}
    if desc is not None: self.serialize_params["desc"] = desc

  def higher_is_better(self): return True

  def score_str(self):
    return "{:.2f}%".format(self.value() * 100.0)

  def value(self):
    return self.recall

  def metric_name(self):
    return "Recall" + str(self.nbest)

  @staticmethod
  def aggregate(scores: Sequence['RecallScore'], desc: Any = None) -> 'RecallScore':
    return RecallScore(recall=np.average(s.recall for s in scores), hyp_len=len(scores), ref_len=len(scores), nbest=scores[0].nbest, desc=desc)


class ExternalScore(models.EvalScore, xnmt.Serializable):
  yaml_tag = "!ExternalScore"
  """
  Class to keep a score computed with an external tool.

  Args:
    value: score value
    higher_is_better: whether higher scores or lower scores are favorable
    desc: human-readable description to include in log outputs
  """
  @xnmt.serializable_init
  def __init__(self, value: float, higher_is_better: bool = True, desc: Any = None) -> None:
    self.value = value
    self.higher_is_better = higher_is_better
    self.desc = desc
    self.serialize_params = {"value":value, "higher_is_better":higher_is_better}
    if desc is not None: self.serialize_params["desc"] = desc
  def value(self): return self.value
  def metric_name(self): return "External"
  def higher_is_better(self): return self.higher_is_better
  def score_str(self):
    return "{:.3f}".format(self.value)


class SequenceAccuracyScore(SentenceLevelEvalScore, xnmt.Serializable):
  yaml_tag = "!SequenceAccuracyScore"
  """
  Class to keep a sequence accuracy score.

  Args:
    num_correct: number of correct outputs
    num_total: number of total outputs
    desc: human-readable description to include in log outputs
  """
  @xnmt.serializable_init
  def __init__(self,
               num_correct: int,
               num_total: int,
               desc: Any = None):
    self.num_correct = num_correct
    self.num_total = num_total
    self.desc = desc
    self.serialize_params = {"num_correct":num_correct, "num_total":num_total}
    if desc is not None: self.serialize_params["desc"] = desc
  def higher_is_better(self): return True
  def value(self): return self.num_correct / self.num_total
  def metric_name(self): return "SequenceAccuracy"
  def score_str(self):
    return f"{self.value()*100.0:.2f}%"

  @staticmethod
  def aggregate(scores: Sequence['SentenceLevelEvalScore'], desc: Any = None):
    return SequenceAccuracyScore(num_correct=sum(s.num_correct for s in scores),
                                 num_total=sum(s.num_total for s in scores),
                                 desc=desc)


class FMeasureScore(SentenceLevelEvalScore, xnmt.Serializable):
  yaml_tag = "!FMeasureScore"
  @xnmt.serializable_init
  def __init__(self,
               true_pos: int,
               false_neg: int,
               false_pos: int,
               desc: Any = None):
    self.true_pos = true_pos
    self.false_neg = false_neg
    self.false_pos = false_pos
    self.serialize_params = {"true_pos": true_pos, "false_neg": false_neg, "false_pos": false_pos}
    if desc is not None: self.serialize_params["desc"] = desc
  def higher_is_better(self): return True
  def value(self):
    if self.true_pos + self.false_neg + self.false_pos > 0:
      return 2*self.true_pos/(2*self.true_pos + self.false_neg + self.false_pos)
    else:
      return "n/a"
  def metric_name(self): return "F1 Score"
  def score_str(self):
    prec = 0
    if self.true_pos+self.false_pos > 0: prec = self.true_pos/(self.true_pos+self.false_pos)
    rec = 0
    if self.true_pos+self.false_neg > 0: rec = self.true_pos/(self.true_pos+self.false_neg)
    val = self.value()
    if isinstance(val, float): val = f"{self.value()*100.0:.2f}%"
    return f"{val} " \
           f"(prec: {prec}, " \
           f"recall: {rec}; " \
           f"TP={self.true_pos},FP={self.false_pos},FN={self.false_neg})"
  @staticmethod
  def aggregate(scores: Sequence['SentenceLevelEvalScore'], desc: Any = None):
    return FMeasure( true_pos=sum(s.true_pos for s in scores),
                    false_neg=sum(s.false_neg for s in scores),
                    false_pos=sum(s.false_pos for s in scores),
                    desc=desc)





