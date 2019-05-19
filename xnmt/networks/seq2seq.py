import dynet as dy

import xnmt
import xnmt.models as models
import xnmt.modules.nn as nn

from typing import List


class Seq2Seq(models.AutoRegressiveModel, xnmt.Serializable):
  yaml_tag = "!Seq2Seq"
  @xnmt.serializable_init
  def __init__(self,
               src_reader: models.InputReader,
               trg_reader: models.InputReader,
               encoder: models.Encoder = xnmt.bare(nn.SeqEncoder),
               decoder: models.Decoder = xnmt.bare(nn.ArbLenDecoder)):
    super().__init__(src_reader, trg_reader)
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
    return self.decoder.add_input(state, word)

  def best_k(self, state: models.UniDirectionalState, k: int, normalize_scores: bool = False) -> List[models.SearchAction]:
    return self.decoder.best_k(state, k, normalize_scores)

  def sample(self, state: models.UniDirectionalState, n: int, temperature: float = 1.0) -> List[models.SearchAction]:
    return self.decoder.sample(state, n, temperature)

  def auto_regressive_states(self, src: xnmt.Batch, trg: List[xnmt.Batch]) -> List[models.UniDirectionalState]:
    decoder_states = []
    dec_state = self.initial_state(src)
    for i in range(len(trg)):
      prev_word = None if i == 0 else trg[i-1]
      dec_state = self.decoder.add_input(dec_state, prev_word)
      decoder_states.append(dec_state)
    return decoder_states

  def hyp_to_readable(self, search_hyps: List[models.Hypothesis], idx: int):
    ret = []
    for search_hyp in search_hyps:
      actions = search_hyp.actions()
      word_ids = [action.action_id for action in actions]

#      Attention is maybe needed in the future?
#      if hasattr(actions[0].decoder_state, "attender_state"):
#        attentions = [action.decoder_state.attender_state.attention for action in actions]
#      else:
#        attentions = None
     
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
      # TODO(philip30): add more for RNNG?
      
      # NBest or not?
      if len(search_hyps) == 1:
        ret.append(sent)
      else:
        ret.append(xnmt.structs.sentences.NbestSentence(sent, idx))
    return ret


#        if self.is_reporting:
#          attentions = np.concatenate([x.npvalue() for x in attentions], axis=1)
#          self.report_sent_info({"attentions": attentions,
#                                 "src": src[0],
#                                 "output": sent})

#    return sent.DepTreeRNNGSequenceSentence(idx=src[0].idx,
#                                            score=score,
#                                            actions=output_actions,
#                                            surface_vocab=getattr(self.trg_reader, "surface_vocab", None),
#                                            nt_vocab=getattr(self.trg_reader, "nt_vocab", None),
#                                            edge_vocab=getattr(self.trg_reader, "edge_vocab", None),
#                                            output_procs=self.trg_reader.output_procs)
