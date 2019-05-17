
from typing import Callable, Sequence, Any, Optional, Union

import xnmt
import xnmt.models as models
import xnmt.eval.evaluators as evaluators

__all__ = ["eval_shortcuts", "evaluate"]

eval_shortcuts = {
  "bleu": lambda: evaluators.BLEUEvaluator(),
  "gleu": lambda: evaluators.GLEUEvaluator(),
  "wer": lambda: evaluators.WEREvaluator(),
  "cer": lambda: evaluators.CEREvaluator(),
  "recall": lambda: evaluators.RecallEvaluator(),
  "accuracy": lambda: evaluators.SequenceAccuracyEvaluator(),
  "fmeasure" : lambda: evaluators.FMeasureEvaluator(),
  "seg_fmeasure": lambda: evaluators.SegmentationFMeasureEvaluator(),
  "rnng_parse_fmeasure": lambda: evaluators.RNNGParseFMeasureEvaluator(),
}

def read_data(loc_: str, post_process: Optional[Callable[[str], str]] = None) -> Sequence[str]:
  """Reads the lines in the file specified in loc_ and return the list after inserting the tokens
  """
  data = list()
  with open(loc_, encoding='utf-8') as fp:
    for line in fp:
      t = line.strip()
      if post_process is not None:
        t = post_process(t)
      data.append(t)
  return data

def evaluate(ref_file: Union[str, Sequence[str]],
             hyp_file: Union[str, Sequence[str]],
             evaluator_list: Sequence[models.Evaluator],
             desc: Any = None) -> Sequence[models.EvalScore]:
  """"Returns the eval score (e.g. BLEU) of the hyp sents using reference trg sents

  Args:
    ref_file: path of the reference file
    hyp_file: path of the hypothesis trg file
    evaluator_list: Evaluation metrics. Can be a list of evaluator objects, or a shortcut string
    desc: descriptive string passed on to evaluators
  """
  hyp_postprocess = lambda line: line.split()
  ref_postprocess = lambda line: line.split()

  is_multi = False
  if isinstance(ref_file, str):
    ref_corpus = read_data(ref_file, post_process=ref_postprocess)
  else:
    if len(ref_file)==1: ref_corpus = read_data(ref_file[0], post_process=ref_postprocess)
    else:
      is_multi = True
      ref_corpora = [read_data(ref_file_i, post_process=ref_postprocess) for ref_file_i in ref_file]
      ref_corpus = [tuple(ref_corpora[i][j] for i in range(len(ref_file))) for j in range(len(ref_corpora[0]))]
  hyp_corpus = read_data(hyp_file, post_process=hyp_postprocess)
  len_before = len(hyp_corpus)
  ref_corpus, hyp_corpus = zip(*filter(lambda x: xnmt.globals.NO_DECODING_ATTEMPTED not in x[1], zip(ref_corpus, hyp_corpus)))
  if len(ref_corpus) < len_before:
    xnmt.logger.info(f"> ignoring {len_before - len(ref_corpus)} out of {len_before} test sentences.")

  if is_multi:
    return [evaluator.evaluate_multi_ref(ref_corpus, hyp_corpus, desc=desc) for evaluator in evaluator_list]
  else:
    return [evaluator.evaluate(ref_corpus, hyp_corpus, desc=desc) for evaluator in evaluator_list]


