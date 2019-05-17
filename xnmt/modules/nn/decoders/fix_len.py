from typing import Optional, Any, List

import xnmt
import xnmt.models as models
import xnmt.modules.nn as nn
import xnmt.modules.nn.decoders.states


class FixLenDecoder(models.Decoder, xnmt.Serializable):
  yaml_tag = "!FixLenDecoder"
  @xnmt.serializable_init
  def __init__(self,
               attender: models.Attender = xnmt.bare(nn.MlpAttender),
               transform: models.Transform = xnmt.bare(nn.NonLinear),
               scorer: models.Scorer = xnmt.bare(nn.Softmax)):
    self.attender = attender
    self.transform = transform
    self.scorer = scorer

  def initial_state(self, enc_results: models.EncoderState):
    attender_state = self.attender.initial_state(enc_results.encode_seq)
    return xnmt.modules.nn.decoders.states.FixSeqLenDecoderState(enc_results.encode_seq, attender_state)

  def add_input(self, dec_state: xnmt.modules.nn.decoders.states.FixSeqLenDecoderState, trg_word: Optional[xnmt.Batch]):
    return xnmt.modules.nn.decoders.states.FixSeqLenDecoderState(dec_state.encodings, dec_state.timestep + 1)

  def calc_loss(self, dec_state: xnmt.modules.nn.decoders.states.FixSeqLenDecoderState, ref_action: xnmt.Batch):
    return self.scorer.calc_loss(self.transform.transform(dec_state.as_vector()), ref_action)

  def best_k(self, dec_state: xnmt.modules.nn.decoders.states.FixSeqLenDecoderState, k: int, normalize_scores=False) -> List[models.SearchAction]:
    best_k = self.scorer.best_k(self.transform.transform(dec_state.as_vector()), k ,normalize_scores)
    return [models.SearchAction(dec_state, best_word, best_score, None) for best_word, best_score in best_k]

  def sample(self, dec_state: xnmt.modules.nn.decoders.states.FixSeqLenDecoderState, n: int, temperature=1.0) -> List[models.SearchAction]:
    sample_k =  self.scorer.sample(self.transform.transform(dec_state.as_vector()), n)
    return [models.SearchAction(dec_state, best_word, best_score, None) for best_word, best_score in sample_k]

  def finish_generating(self, dec_output: Any, dec_state: xnmt.modules.nn.decoders.states.FixSeqLenDecoderState):
    return dec_state.timestep == xnmt.globals.singleton_global.src_batch[0].len_unpadded()