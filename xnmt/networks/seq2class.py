import xnmt
import xnmt.models as models
import xnmt.modules.nn as nn


class Seq2Class(models.ConditionedModel, models.GeneratorModel, xnmt.Serializable):
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
               encoder: models.Encoder = xnmt.bare(nn.SentenceEncoder),
               inference: models.Inference = xnmt.bare(xnmt.inferences.IndependentOutputInference),
               transform: models.Transform = xnmt.bare(nn.NonLinear),
               scorer: models.Scorer = xnmt.bare(nn.Softmax)):
    models.GeneratorModel.__init__(self)

    super().__init__()
    self.encoder = encoder
    self.transform = transform
    self.scorer = scorer
    self.inference = inference

  def calc_nll(self, src: xnmt.Batch, trg: xnmt.Batch) -> xnmt.LossExpr:
    units = [t.len_unpadded() for t in trg]
    ids = xnmt.mark_as_batch([t.value for t in trg])
    h = self.initial_state(src)
    loss_expr = self.scorer.calc_loss(h.as_vector(), ids)
    return xnmt.LossExpr(loss_expr, units)

  def generate_single(self, src: xnmt.Batch, search_strategy: 'xnmt.models.SearchStrategy', is_sort: bool=True):
    outputs = []
    for batch_i in range(src.batch_size()):
      if src.batch_size() > 1:
        word = best_words[0, batch_i]
        score = best_scores[0, batch_i]
      else:
        word = best_words[0]
        score = best_scores[0]
      outputs.append(xnmt.structs.sentences.ScalarSentence(value=word, score=score))

  def initial_state(self, src: xnmt.Batch) -> models.DecoderState:
    xnmt.event_trigger.start_sent(src)
    encoding_result = self.encoder.encode(src)
    h = encoding_result.encoder_final_states[-1].main_expr()
    return models.ClassifierState(self.transform.transform(h))

  def best_k(self, dec_state: models.DecoderState, k: int, normalize_scores: bool):
    return self.scorer.best_k(dec_state.as_vector(), k, normalize_scores)

  def sample(self, dec_state: models.DecoderState, k: int):
    raise self.scorer.sample(dec_state.as_vector(), k)


