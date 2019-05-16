import yaml
import math
import numpy as np
import subprocess

from collections import Counter, defaultdict
from xnmt.eval.metrics import *
from typing import Optional, Sequence, Union, Any, Dict, List, Tuple

import xnmt
import xnmt.models as models


class SentenceLevelEvaluator(models.Evaluator):
  """
  A template class for sentence-level evaluators.

  Args:
    write_sentence_scores: path of file to write sentence-level scores to (in YAML format)
  """
  def __init__(self, write_sentence_scores:Optional[str] = None):
    self.write_sentence_scores = write_sentence_scores

  def evaluate(self, ref: Sequence, hyp: Sequence, desc: Any = None) -> SentenceLevelEvalScore:
    assert (len(ref) == len(hyp)), \
      "Length of reference corpus and candidate corpus must be same"
    sentence_scores = [self.evaluate_one_sent(ref_i, hyp_i) for (ref_i,hyp_i) in zip(ref,hyp)]
    if self.write_sentence_scores:
      with open(self.write_sentence_scores, "w") as f_out: f_out.write(yaml.dump(sentence_scores))
    return sentence_scores[0].__class__.aggregate(sentence_scores, desc=desc)

  def evaluate_one_sent(self, ref: Any, hyp: Any) -> SentenceLevelEvalScore:
    raise NotImplementedError("evaluate_one_sent must be implemented in SentenceLevelEvaluator subclasses")

  def evaluate_multi_ref(self, ref: Sequence[Sequence], hyp: Sequence, desc: Any = None) -> models.EvalScore:
    sentence_scores = []
    for ref_alternatives_i, hyp_i in zip(ref, hyp):
      cur_best = None
      for ref_ij in ref_alternatives_i:
        cur_score = self.evaluate_one_sent(ref_ij, hyp_i)
        if cur_best is None or cur_score.better_than(cur_best):
          cur_best = cur_score
      sentence_scores.append(cur_best)
    if self.write_sentence_scores:
      with open(self.write_sentence_scores, "w") as f_out: f_out.write(yaml.dump(sentence_scores))
    return sentence_scores[0].__class__.aggregate(sentence_scores, desc=desc)


class FastBLEUEvaluator(SentenceLevelEvaluator, xnmt.Serializable):
  yaml_tag = "!FastBLEUEvaluator"
  """
  Class for computing BLEU scores using a fast Cython implementation.

  Does not support multiple references.
  BLEU scores are computed according to K Papineni et al "BLEU: a method for automatic evaluation of machine translation"

  Args:
    ngram: consider ngrams up to this order (usually 4)
    smooth:
  """
  @xnmt.serializable_init
  def __init__(self, ngram: int = 4, smooth: float = 1, write_sentence_scores: Optional[str] = None):
    super().__init__(write_sentence_scores)
    self.ngram = ngram
    self.weights = (1 / ngram) * np.ones(ngram, dtype=np.float32)
    self.smooth = smooth
    self.reference_corpus = None
    self.candidate_corpus = None

  def evaluate_one_sent(self, ref, hyp):
    try:
      from xnmt.cython import xnmt_cython
    except:
      xnmt.logger.error("BLEU evaluate fast requires xnmt cython installation step."
                   "please check the documentation.")
      raise
    if len(ref) == 0 or len(hyp) == 0: return 0
    return xnmt_cython.bleu_sentence(self.ngram, self.smooth, ref, hyp)


class BLEUEvaluator(models.Evaluator, xnmt.Serializable):
  yaml_tag = "!BLEUEvaluator"
  """
  Compute BLEU scores against one or several references.

  BLEU scores are computed according to K Papineni et al "BLEU: a method for automatic evaluation of machine translation"

  Args:
    ngram: consider ngrams up to this order (usually 4)
  """
  @xnmt.serializable_init
  def __init__(self, ngram: int = 4):
    self.ngram = ngram
    self.weights = (1 / ngram) * np.ones(ngram, dtype=np.float32)
    self.reference_corpus = None
    self.candidate_corpus = None

  def evaluate(self, ref: Sequence[Sequence[str]], hyp: Sequence[Sequence[str]], desc: Any = None) -> BLEUScore:
    """
    Args:
      ref: reference sentences (single-reference case: sentence is list of strings;
      hyp: list of hypothesis sentences ( a sentence is a list of tokens )
      desc: description to pass on to returned score
    Return:
      Score, including intermediate results such as ngram ratio, sentence length, brevity penalty
    """
    return self._eval(ref, hyp, is_multi_ref=False, desc=desc)

  def evaluate_multi_ref(self, ref: Sequence[Sequence[Sequence[str]]], hyp: Sequence[Sequence[str]],
                         desc: Any = None) -> BLEUScore:
    """
    Args:
      ref: list of tuples of reference sentences ( a sentence is a list of tokens )
      hyp: list of hypothesis sentences ( a sentence is a list of tokens )
      desc: optional description that is passed on to score objects
    Return:
      Score, including intermediate results such as ngram ratio, sentence length, brevity penalty
    """
    return self._eval(ref, hyp, is_multi_ref=True, desc=desc)

  def _eval(self, ref: Sequence[Union[Sequence[str], Sequence[Sequence[str]]]], hyp: Sequence[Sequence[str]],
            is_multi_ref: bool, desc: Any = None) -> BLEUScore:
    self.reference_corpus = ref
    self.candidate_corpus = hyp

    assert (len(self.reference_corpus) == len(self.candidate_corpus)), \
           "Length of Reference Corpus and Candidate Corpus should be same"

    # Modified Precision Score
    clipped_ngram_count = Counter()
    candidate_ngram_count = Counter()

    # Brevity Penalty variables
    word_counter = Counter()

    for ref_sent, can_sent in zip(self.reference_corpus, self.candidate_corpus):
      word_counter['candidate'] += len(can_sent)
      if not is_multi_ref:
        word_counter['reference'] += len(ref_sent)

        clip_count_dict, full_count_dict = self._modified_precision(ref_sent, can_sent)

      else:
        ref_lens = sorted([(len(ref_sent_i), abs(len(ref_sent_i) - len(can_sent))) for ref_sent_i in ref_sent],
                          key=lambda x: (x[1],x[0]))
        word_counter['reference'] += ref_lens[0][0]
        counts = [self._modified_precision(ref_sent_i, can_sent) for ref_sent_i in ref_sent]
        full_count_dict = counts[0][1]
        clip_count_dict = defaultdict(Counter)
        for ngram_type in candidate_ngram_count:
          for i in range(len(counts)):
            clip_count_dict[ngram_type] |= counts[i][0][ngram_type]

      for ngram_type in full_count_dict:
        if ngram_type in clip_count_dict:
          clipped_ngram_count[ngram_type] += sum(clip_count_dict[ngram_type].values())
        candidate_ngram_count[ngram_type] += sum(full_count_dict[ngram_type].values())

    # Edge case
    # Return 0 if there are no matching n-grams
    # If there are no unigrams, return BLEU score of 0
    # No need to check for higher order n-grams
    if clipped_ngram_count[1] == 0:
      return BLEUScore(bleu=None, ngram=self.ngram, desc=desc)

    frac_score_list = list()
    log_precision_score = 0.
    # Precision Score Calculation
    for ngram_type in range(1, self.ngram + 1):
      frac_score = 0
      if clipped_ngram_count[ngram_type] == 0:
        log_precision_score += -1e10
      else:
        frac_score = clipped_ngram_count[ngram_type] / candidate_ngram_count[ngram_type]
        log_precision_score += self.weights[ngram_type - 1] * math.log(frac_score)
      frac_score_list.append("%.6f" % frac_score)

    precision_score = math.exp(log_precision_score)

    # Brevity Penalty Score
    brevity_penalty_score = self._brevity_penalty(word_counter['reference'], word_counter['candidate'])

    # BLEU Score
    bleu_score = brevity_penalty_score * precision_score
    return BLEUScore(bleu_score, frac_score_list, brevity_penalty_score, word_counter['candidate'], word_counter['reference'], ngram=self.ngram, desc=desc)

  def _brevity_penalty(self, r: int, c: int) -> float:
    """
    Args:
      r: number of words in reference corpus
      c: number of words in candidate corpus
    Return:
      brevity penalty score
    """

    penalty = 1.

    # If candidate sent length is 0 (empty), return 0.
    if c == 0:
      return 0.
    elif c <= r:
      penalty = np.exp(1. - (r / c))
    return penalty

  def _extract_ngrams(self, tokens: Sequence[str]) -> Dict[int, Counter]:
    """
    Extracts ngram counts from the input string

    Args:
      tokens: tokens of string for which the ngram is to be computed
    Return:
      a Counter object containing ngram counts
    """

    ngram_count = defaultdict(Counter)
    num_words = len(tokens)

    for i, first_token in enumerate(tokens[0: num_words]):
      for j in range(0, self.ngram):
        outer_range = i + j + 1
        ngram_type = j + 1
        if outer_range <= num_words:
          ngram_tuple = tuple(tokens[i: outer_range])
          ngram_count[ngram_type][ngram_tuple] += 1

    return ngram_count

  def _modified_precision(self, reference_sent: Sequence[str], candidate_sent: Sequence[str]) \
          -> Tuple[Dict[int,Counter],Dict[int,Counter]]:
    """
    Computes counts useful in modified precision calculations

    Args:
      reference_sent: iterable of tokens
      candidate_sent: iterable of tokens
    Return: tuple of Counter objects
    """

    clipped_ngram_count = defaultdict(Counter)

    reference_ngram_count = self._extract_ngrams(reference_sent)
    candidate_ngram_count = self._extract_ngrams(candidate_sent)

    for ngram_type in candidate_ngram_count:
      clipped_ngram_count[ngram_type] = candidate_ngram_count[ngram_type] & reference_ngram_count[ngram_type]

    return clipped_ngram_count, candidate_ngram_count


class GLEUEvaluator(SentenceLevelEvaluator, xnmt.Serializable):
  yaml_tag = "!GLEUEvaluator"
  """
  Class for computing GLEU (Google BLEU) Scores.

  GLEU scores are described in https://arxiv.org/pdf/1609.08144v2.pdf as follows:

        "The BLEU score has some undesirable properties when used for single
        sentences, as it was designed to be a corpus measure. We therefore
        use a slightly different score for our RL experiments which we call
        the 'GLEU score'. For the GLEU score, we record all sub-sequences of
        1, 2, 3 or 4 tokens in output and target sequence (n-grams). We then
        compute a recall, which is the ratio of the number of matching n-grams
        to the number of total n-grams in the target (ground truth) sequence,
        and a precision, which is the ratio of the number of matching n-grams
        to the number of total n-grams in the generated output sequence. Then
        GLEU score is simply the minimum of recall and precision. This GLEU
        score's range is always between 0 (no matches) and 1 (all match) and
        it is symmetrical when switching output and target. According to
        our experiments, GLEU score correlates quite well with the BLEU
        metric on a corpus level but does not have its drawbacks for our per
        sentence reward objective."

  Args:
    min_length: minimum n-gram order to consider
    max_length: maximum n-gram order to consider
    write_sentence_scores: path of file to write sentence-level scores to (in YAML format)
  """
  @xnmt.serializable_init
  def __init__(self,
               min_length: int = 1,
               max_length: int = 4,
               write_sentence_scores: Optional[str] = None) -> None:
    super().__init__(write_sentence_scores=write_sentence_scores)
    self.min = min_length
    self.max = max_length

  def _extract_all_ngrams(self, tokens):
    """
    Extracts ngram counts from the input string

    Args:
      tokens: tokens of string for which the ngram is to be computed
    Return:
      a Counter object containing ngram counts for self.min <= n <= self.max
    """
    num_words = len(tokens)
    ngram_count = Counter()
    for i, first_token in enumerate(tokens[0: num_words]):
      for n in range(self.min, self.max + 1):
        outer_range = i + n
        if outer_range <= num_words:
          ngram_tuple = tuple(tokens[i: outer_range])
          ngram_count[ngram_tuple] += 1
    return ngram_count

  def evaluate_one_sent(self, ref:Sequence[str], hyp:Sequence[str]):
    """
    Args:
      ref: reference sentence ( a sent is a list of tokens )
      hyp: hypothesis sentence ( a sent is a list of tokens )
    Return:
      GLEU score object
    """
    hyp_ngrams = self._extract_all_ngrams(hyp)
    tot_ngrams_hyp = sum(hyp_ngrams.values())
    ref_ngrams = self._extract_all_ngrams(ref)
    tot_ngrams_ref = sum(ref_ngrams.values())

    overlap_ngrams = ref_ngrams & hyp_ngrams
    n_match = sum(overlap_ngrams.values())
    n_total = max(tot_ngrams_hyp, tot_ngrams_ref)

    return GLEUScore(n_match, n_total, hyp_len = len(hyp), ref_len = len(ref))


class WEREvaluator(SentenceLevelEvaluator, xnmt.Serializable):
  yaml_tag = "!WEREvaluator"
  """
  A class to evaluate the quality of output in terms of word error rate.

  Args:
    case_sensitive: whether scoring should be case-sensitive
    write_sentence_scores: path of file to write sentence-level scores to (in YAML format)
  """
  @xnmt.serializable_init
  def __init__(self, case_sensitive: bool = False, write_sentence_scores: Optional[str] = None):
    super().__init__(write_sentence_scores=write_sentence_scores)
    self.case_sensitive = case_sensitive
    self.aligner = xnmt.modules.levenshtein.LevenshteinAligner()

  def evaluate_one_sent(self, ref: Sequence[str], hyp: Sequence[str]) -> WERScore:
    if not self.case_sensitive:
      hyp = [w.lower() for w in hyp]
      ref = [w.lower() for w in ref]
    _,_,_,alignment = self.aligner.align(ref, hyp)

    score = WERScore(correct=len([a for a in alignment if a=='c']),
                     substitutions=len([a for a in alignment if a == 's']),
                     insertions=len([a for a in alignment if a == 'i']),
                     deletions=len([a for a in alignment if a == 'd']))
    assert score.ref_len() == len(ref)
    assert score.hyp_len() == len(hyp)
    return score


class CEREvaluator(SentenceLevelEvaluator, xnmt.Serializable):
  yaml_tag = "!CEREvaluator"
  """
  A class to evaluate the quality of output in terms of character error rate.

  Args:
    case_sensitive: whether scoring should be case-sensitive
    write_sentence_scores: path of file to write sentence-level scores to (in YAML format)
  """
  @xnmt.serializable_init
  def __init__(self, case_sensitive: bool = False, write_sentence_scores: Optional[str] = None):
    super().__init__(write_sentence_scores=write_sentence_scores)
    self.case_sensitive = case_sensitive
    self.aligner = xnmt.modules.levenshtein.LevenshteinAligner()

  def evaluate_one_sent(self, ref: Sequence[str], hyp: Sequence[str]) -> CERScore:
    """
    Calculate the quality of output sentence given a reference.

    Args:
      ref: list of reference words
      hyp: list of decoded words
    Return:
      character error rate: (ins+del+sub) / (ref_len)
    """
    ref_char = list("".join(ref))
    hyp_char = list("".join(hyp))
    if not self.case_sensitive:
      hyp_char = [w.lower() for w in hyp_char]
      ref_char = [w.lower() for w in ref_char]
    _,_,_,alignment = self.aligner.align(ref_char, hyp_char)
    score = CERScore(correct=len([a for a in alignment if a=='c']),
                     substitutions=len([a for a in alignment if a == 's']),
                     insertions=len([a for a in alignment if a == 'i']),
                     deletions=len([a for a in alignment if a == 'd']))
    assert score.ref_len() == len(ref_char)
    assert score.hyp_len() == len(hyp_char)
    return score


class ExternalEvaluator(models.Evaluator, xnmt.Serializable):
  yaml_tag = "!ExternalEvaluator"
  """
  A class to evaluate the quality of the output according to an external evaluation script.

  Does not support multiple references.
  The external script should only print a number representing the calculated score.

  Args:
    path: path to external command line tool.
    higher_better: whether to interpret higher scores as favorable.
  """
  @xnmt.serializable_init
  def __init__(self, path:str=None, higher_better:bool=True):
    self.path = path
    self.higher_better = higher_better

  def evaluate(self, ref, hyp, desc=None):
    """
    Calculate the quality of output according to an external script.

    Args:
      ref: (ignored)
      hyp: (ignored)
      desc: description to pass on to returned score
    Return:
      external eval script score
    """
    proc = subprocess.Popen([self.path], stdout=subprocess.PIPE, shell=True)
    (out, _) = proc.communicate()
    external_score = float(out)
    return ExternalScore(external_score, higher_is_better=self.higher_better, desc=desc)


class RecallEvaluator(SentenceLevelEvaluator, xnmt.Serializable):
  yaml_tag = "!RecallEvaluator"
  """
  Compute recall by counting true positives.

  Args:
    nbest: compute recall within n-best of specified n
    write_sentence_scores: path of file to write sentence-level scores to (in YAML format)
  """
  @xnmt.serializable_init
  def __init__(self, nbest: int = 5, write_sentence_scores: Optional[str] = None):
    super().__init__(write_sentence_scores=write_sentence_scores)
    self.nbest = nbest

  def evaluate(self, ref, hyp, desc=None):
    true_positive = 0
    for hyp_i, ref_i in zip(hyp, ref):
      if any(ref_i == idx for idx, _ in hyp_i[:self.nbest]):
        true_positive += 1
    score = true_positive / float(len(ref))
    return RecallScore(score, len(hyp), len(ref), nbest=self.nbest, desc=desc)

  def evaluate_one_sent(self, ref:Any, hyp:Any):
    score = 1.0 if any(ref == idx for idx, _ in hyp[:self.nbest]) else 0.0
    return RecallScore(score, hyp_len=1, ref_len=1, nbest=self.nbest)


class SequenceAccuracyEvaluator(SentenceLevelEvaluator, xnmt.Serializable):
  yaml_tag = "!SequenceAccuracyEvaluator"
  """
  A class to evaluate the quality of output in terms of sequence accuracy.

  Args:
    case_sensitive: whether differences in capitalization are to be considered
    write_sentence_scores: path of file to write sentence-level scores to (in YAML format)
  """
  @xnmt.serializable_init
  def __init__(self, case_sensitive=False, write_sentence_scores: Optional[str] = None) -> None:
    super().__init__(write_sentence_scores=write_sentence_scores)
    self.case_sensitive = case_sensitive

  def _compare(self, ref_sent, hyp_sent):
    if not self.case_sensitive:
      hyp_sent = [w.lower() for w in hyp_sent]
    if not self.case_sensitive:
      ref_sent = [w.lower() for w in ref_sent]
    return ref_sent == hyp_sent

  def evaluate_one_sent(self, ref:Sequence[str], hyp:Sequence[str]):
    """
    Calculate the accuracy of output given a references.

    Args:
      ref: list of list of reference words
      hyp: list of list of decoded words
    Return: formatted string
    """
    correct = 1 if self._compare(ref, hyp) else 0
    return SequenceAccuracyScore(num_correct=correct, num_total=1)


class FMeasureEvaluator(SentenceLevelEvaluator, xnmt.Serializable):
  yaml_tag = "!FMeasureEvaluator"
  """
  A class to evaluate the quality of output in terms of classification F-score.

  Args:
    pos_token: token for the 'positive' class
    write_sentence_scores: path of file to write sentence-level scores to (in YAML format)
  """
  @xnmt.serializable_init
  def __init__(self, pos_token:str="1", write_sentence_scores: Optional[str] = None) -> None:
    super().__init__(write_sentence_scores=write_sentence_scores)
    self.pos_token = pos_token

  def evaluate_one_sent(self, ref:Sequence[str], hyp:Sequence[str]):
    """
    Calculate the accuracy of output given a references.

    Args:
      ref: list of list of reference words
      hyp: list of list of decoded words
    Return: formatted string
    """
    if len(ref)!=1 or len(hyp)!=1: raise ValueError("FScore requires scalar ref and hyp")
    ref = ref[0]
    hyp = hyp[0]
    return FMeasureScore( true_pos=1 if (ref == hyp) and (hyp == self.pos_token) else 0,
                          false_neg=1 if (ref != hyp) and (hyp != self.pos_token) else 0,
                          false_pos=1 if (ref != hyp) and (hyp == self.pos_token) else 0)


class SegmentationFMeasureEvaluator(SentenceLevelEvaluator, xnmt.Serializable):
  yaml_tag = "!SegmentationFMeasureEvaluator"
  @xnmt.serializable_init
  def __init__(self, write_sentence_scores: Optional[str] = None) -> None:
    super().__init__(write_sentence_scores=write_sentence_scores)

  def evaluate_one_sent(self, ref:Sequence[str], hyp:Sequence[str]):
    hyp = [x.replace("<unk>","_") for x in hyp]

    hyp_seg = [len(x) for x in hyp]
    ref_seg = [len(x) for x in ref]
    hyp_sum = sum(hyp_seg)
    ref_sum = sum(ref_seg)

    assert hyp_sum == ref_sum, \
           "Bad Line {} != {}: \n{}\n{}".format(hyp_sum, ref_sum, " ".join(hyp), " ".join(ref))

    hyp_dec = [0 for _ in range(hyp_sum)]
    ref_dec = [0 for _ in range(ref_sum)]

    position = 0
    for seg in hyp_seg:
      position += seg
      hyp_dec[position-1] = 1
    position = 0
    for seg in ref_seg:
      position += seg
      ref_dec[position-1] = 1

    tp, fn, fp = 0, 0, 0
    for pred, act in zip(hyp_dec, ref_dec):
      if pred == act:
        tp += 1
      elif pred == 1:
        fp += 1
      else:
        fn += 1
    return FMeasureScore(true_pos=tp, false_neg=fn, false_pos=fp)


class RNNGParseFMeasureEvaluator(SentenceLevelEvaluator, xnmt.Serializable):
  yaml_tag = "!RNNGParseFMeasureEvaluator"
  @xnmt.serializable_init
  def __init__(self, ignore_word_in_gen=False, write_sentence_scores: Optional[str]=None):
    super().__init__(write_sentence_scores=write_sentence_scores)
    self.ignore_word_in_gen = ignore_word_in_gen
    self.gleu_evalutor = GLEUEvaluator()

  def evaluate_one_sent(self, ref: List[str], hyp: List[str]):
    if self.ignore_word_in_gen:
      for i in range(len(ref)):
        if ref[i].startswith("GEN"):
          ref[i] = "SHIFT"
      for i in range(len(hyp)):
        if hyp[i].startswith("GEN"):
          hyp[i] = "SHIFT"

    return self.gleu_evalutor.evaluate_one_sent(ref, hyp)

