"""
This module contains features related to outputs generated by a model.

The main responsibilities are data structures for holding such outputs, and code to translate outputs into readable
strings.
"""
from typing import Union, List, Optional

import xnmt
import xnmt.models as models


def get_output_processor(spec: Optional[Union[str, List[models.OutputProcessor]]]) -> List[models.OutputProcessor]:
  if isinstance(spec, str):
    procs = []
    for spec_item in spec.split(","):
      if spec_item == "none":
        continue
      elif spec_item == "join-char":
        procs.append(JoinCharTextOutputProcessor())
      elif spec == "join-bpe":
        procs.append(JoinBpeTextOutputProcessor())
      elif spec == "join-piece":
        procs.append(JoinPieceTextOutputProcessor())
    return procs
  else:
    return spec


class JoinCharTextOutputProcessor(models.OutputProcessor, xnmt.Serializable):
  """
  Assumes a single-character vocabulary and joins them to form words.

  Per default, double underscores '__' are treated as word separating tokens.
  """
  @xnmt.serializable_init
  def __init__(self, space_token: str = "__") -> None:
    self.space_token = space_token

  def process(self, s: str) -> str:
    return s.replace(" ", "").replace(self.space_token, " ")


class JoinBpeTextOutputProcessor(models.OutputProcessor, xnmt.Serializable):
  """
  Assumes a bpe-based vocabulary and outputs the merged words.

  Per default, the '@' postfix indicates subwords that should be merged
  """
  @xnmt.serializable_init
  def __init__(self, merge_indicator: str = "@@") -> None:
    self.merge_indicator_with_space = merge_indicator + " "

  def process(self, s: str) -> str:
    return s.replace(self.merge_indicator_with_space, "")


class JoinPieceTextOutputProcessor(models.OutputProcessor, xnmt.Serializable):
  """
  Assumes a sentence-piece vocabulary and joins them to form words.

  Space_token could be the starting character of a piece per default, the u'\u2581' indicates spaces
  """
  @xnmt.serializable_init
  def __init__(self, space_token: str = "\u2581") -> None:
    self.space_token = space_token

  def process(self, s: str) -> str:
    return s.replace(" ", "").replace(self.space_token, " ").strip()


class DependencyLeavesOutputProcessor(models.OutputProcessor, xnmt.Serializable):
  @xnmt.serializable_init
  def __init__(self, string_processor: models.OutputProcessor = None):
    self.string_processor = string_processor

  def process(self, rnng_sent) -> str:
    tokens = [rnng_sent.value_vocab[node.value] for node in self._get_nodes(rnng_sent.graph)]
    i = -1
    while -(i+1) < len(tokens) and tokens[i] == xnmt.Vocab.ES_STR:
      i += 1
    tokens = tokens[:-(i+1)]
    sent_str = " ".join(tokens)
    if self.string_processor is not None:
      return self.string_processor.process(sent_str)
    else:
      return sent_str

  def _get_nodes(self, graph):
    return graph.iter_nodes()


class SyntaxLeavesOutputProcessor(DependencyLeavesOutputProcessor, xnmt.Serializable):
  @xnmt.serializable_init
  def __init__(self, string_processor: models.OutputProcessor = None):
    super().__init__(string_processor)

  def _get_nodes(self, graph):
    return [graph[x] for x in graph.leaves()]
