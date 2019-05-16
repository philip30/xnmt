from typing import Optional, Any

import xnmt
import xnmt.models as models
import xnmt.modules.nn as nn


class LabelEmitter(models.Decoder, xnmt.Serializable):
  @xnmt.serializable_init
  def __init__(self,
               transform: models.Transform = xnmt.bare(nn.NonLinear),
               scorer: models.Scorer = xnmt.bare(nn.Softmax)):
    self.transform = transform
    self.scorer = scorer
  
  def initial_state(self, enc_results: models.EncoderState):
    return models.LabelerState(enc_results.encode_seq)
  
  def add_input(self, dec_state: models.LabelerState, trg_word: Optional[xnmt.Batch]):
    return models.LabelerState(dec_state.encodings, dec_state.timestep+1)
  
  def calc_loss(self, dec_state: models.LabelerState, ref_action: xnmt.Batch):
    return self.scorer.calc_loss(self.transform.transform(dec_state.as_vector()), ref_action)
  
  def best_k(self, dec_state: models.LabelerState, k: int, normalize_scores=False):
    return self.scorer.best_k(self.transform.transform(dec_state.as_vector()), k ,normalize_scores)
  
  def sample(self, dec_state: models.LabelerState, n: int, temperature=1.0):
    return self.scorer.sample(self.transform.transform(dec_state.as_vector()), n)
  
  def finish_generating(self, dec_output: Any, dec_state: models.LabelerState):
    return dec_state.timestep == xnmt.globals.singleton_global.src_batch[0].len_unpadded()
