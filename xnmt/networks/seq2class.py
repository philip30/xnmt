import xnmt
import xnmt.models as models
import xnmt.modules.nn as nn

from typing import List

import xnmt.modules.nn.decoders.unidir_states


# TODO Fix This
class Seq2Class(models.ConditionedModel, models.GeneratorModel, models.AutoRegressiveModel, xnmt.Serializable):
  yaml_tag = "!Seq2Class"
  """
  A sequence classifier.

  Runs embeddings through an encoder, feeds the average over all encoder outputs to a transform and scoring layer.

  Args:
    encoder: An encoder to generate encoded inputs
    inference: how to perform inference
    transform: A transform performed before the scoring function
    scorer: A scoring function over the multiple choices
  """
  @xnmt.serializable_init
  def __init__(self,
               encoder: models.Encoder = xnmt.bare(nn.SeqEncoder),
               transform: models.Transform = xnmt.bare(nn.NonLinear),
               scorer: models.Scorer = xnmt.bare(nn.Softmax)):
    models.GeneratorModel.__init__(self)

    super().__init__()
    self.encoder = encoder
    self.transform = transform
    self.scorer = scorer

  def calc_nll(self, src: xnmt.Batch, trg: xnmt.Batch) -> xnmt.LossExpr:
    units = [t.len_unpadded() for t in trg]
    ids = xnmt.mark_as_batch([t.value for t in trg])
    h = self.initial_state(src).encodings[0]
    loss_expr = self.scorer.calc_loss(h.output(), ids)
    return xnmt.LossExpr(loss_expr, units)

  def hyp_to_readable(self, search_hyps: List[models.Hypothesis], idx: int) \
      -> List[xnmt.structs.sentences.ReadableSentence]:
    outputs = []
    for search_hyp in search_hyps:
      actions = search_hyp.actions()
      assert len(actions) == 1
      word = actions[0].action_id
      score = actions[0].log_likelihood
      outputs.append(xnmt.structs.sentences.ScalarSentence(value=word, score=score))
    return outputs

  def add_input(self, inp: xnmt.Batch, state: xnmt.modules.nn.decoders.unidir_states.FixSeqLenUniDirectionalState):
    return xnmt.modules.nn.decoders.unidir_states.FixSeqLenUniDirectionalState(state.encodings, state.timestep + 1)

  def finish_generating(self, output: xnmt.Batch, dec_state: xnmt.modules.nn.decoders.unidir_states.FixSeqLenUniDirectionalState):
    assert dec_state.timestep <= 1
    return dec_state.timestep == 1

  def initial_state(self, src: xnmt.Batch) -> xnmt.modules.nn.decoders.unidir_states.FixSeqLenUniDirectionalState:
    xnmt.event_trigger.start_sent(src)
    encoding_result = self.encoder.encode(src)
    h = encoding_result.encoder_final_states[-1].main_expr()
    return xnmt.modules.nn.decoders.unidir_states.FixSeqLenUniDirectionalState([self.transform.transform(h)])

  def best_k(self, dec_state: models.UniDirectionalState, k: int, normalize_scores: bool):
    return self.scorer.best_k(dec_state.output(), k, normalize_scores)

  def sample(self, dec_state: models.UniDirectionalState, k: int):
    raise self.scorer.sample(dec_state.output(), k)


