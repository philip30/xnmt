import dynet as dy

import xnmt
import xnmt.models as models
import xnmt.modules.nn as nn

from typing import List, Optional


class Seq2Seq(models.AutoRegressiveModel, xnmt.Serializable, models.Reportable):
  yaml_tag = "!Seq2Seq"
  @xnmt.serializable_init
  def __init__(self,
               src_reader: models.InputReader,
               trg_reader: models.InputReader,
               encoder: models.Encoder = xnmt.bare(nn.SeqEncoder),
               decoder: models.Decoder = xnmt.bare(nn.ArbLenDecoder)):
    super().__init__(src_reader, trg_reader)
    models.Reportable.__init__(self)
    self.encoder = encoder
    self.decoder = decoder

  def initial_state(self, src: xnmt.Batch) -> models.UniDirectionalState:
    return self.decoder.initial_state(self.encoder.encode(src), src)

  def finish_generating(self, output: int, dec_state: models.UniDirectionalState):
    return self.decoder.finish_generating(output, dec_state)

  def calc_nll(self, src: xnmt.Batch, trg: xnmt.Batch) -> xnmt.LossExpr:
    ref_words = [xnmt.mark_as_batch([single_trg[i] for single_trg in trg]) for i in range(trg.sent_len())]
    # Encode the sentence
    cur_losses = []
    dec_states = self.auto_regressive_states(src, ref_words)
    for i in range(trg.sent_len()):
      word_loss = self.decoder.calc_loss(dec_states[i], ref_words[i])
      if trg.mask is not None:
        word_loss = trg.mask.cmult_by_timestep_expr(word_loss, i, inverse=True)
      cur_losses.append(word_loss)
    return xnmt.LossExpr(dy.esum(cur_losses), [t.len_unpadded() for t in trg])

  def add_input(self, word, state: models.UniDirectionalState) -> models.UniDirectionalState:
    if word is not None and not xnmt.is_batched(word):
      word = xnmt.mark_as_batch([word])
    return self.decoder.add_input(state, word)

  def best_k(self, state: models.UniDirectionalState, k: int, normalize_scores: bool = False) -> List[models.SearchAction]:
    return self.decoder.best_k(state, k, normalize_scores)

  def sample(self, state: models.UniDirectionalState, n: int, temperature: float = 1.0) -> List[models.SearchAction]:
    return self.decoder.sample(state, n, temperature)

  def pick_oracle(self, state: models.UniDirectionalState, oracle):
    return self.decoder.pick_oracle(oracle, state)

  def auto_regressive_states(self, src: xnmt.Batch, trg: List[xnmt.Batch]) -> List[models.UniDirectionalState]:
    decoder_states = []
    dec_state = self.initial_state(src)
    for i in range(len(trg)):
      prev_word = None if i == 0 else trg[i-1]
      dec_state = self.decoder.add_input(dec_state, prev_word)
      decoder_states.append(dec_state)
    return decoder_states

  def create_trajectories(self, src: xnmt.Batch, search_strategy: 'xnmt.models.SearchStrategy', ref: Optional[xnmt.Batch]=None):
    ret = super().create_trajectories(src, search_strategy, ref)

    if self.is_reporting():
      for src_i, search_hyps in zip(src, ret):
        for i, search_hyp in enumerate(search_hyps):
          self.report_sent_info({"hyp": search_hyps, "src": src_i, "hyp_num": i})

    return ret

  def hyp_to_readable(self, search_hyps: List[models.Hypothesis], idx: int):
    ret = []
    for search_hyp in search_hyps:
      actions = search_hyp.actions()
      word_ids = [action.action_id[0] if isinstance(action.action_id, list) else action.action_id for action in actions]

      if isinstance(self.decoder, nn.ArbLenDecoder):
        sent = xnmt.structs.sentences.SimpleSentence(
          word_ids,
          idx,
          vocab=getattr(self.trg_reader, "vocab", None),
          output_procs=self.trg_reader.output_procs,
          score=search_hyp.score
        )
      else:
        raise NotImplementedError()
      # NBest or not?
      if len(search_hyps) == 1:
        ret.append(sent)
      else:
        ret.append(xnmt.structs.sentences.NbestSentence(sent, idx))
    return ret

