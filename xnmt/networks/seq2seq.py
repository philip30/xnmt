import dynet as dy

import xnmt
import xnmt.models as models
import xnmt.modules.nn as nn

class Seq2SeqModel(models.ConditionedModel, models.GeneratorModel,
                   models.AutoRegressiveModel,
                   xnmt.Serializable):
  @xnmt.serializable_init
  def __init__(self,
               src_reader: models.InputReader = xnmt.ref_src_reader,
               trg_reader: models.InputReader = xnmt.ref_trg_reader,
               encoder: models.Encoder = xnmt.bare(nn.SentenceEncoder),
               decoder: models.Decoder = xnmt.bare(nn.AutoRegressiveDecoder)):
    super().__init__(src_reader, trg_reader)
    self.encoder = encoder
    self.decoder = decoder

  def initial_state(self, src: xnmt.Batch) -> models.DecoderState:
    return self.decoder.initial_state(self.encoder.encode(src))

  def finish_generating(self, output: int, dec_state: models.DecoderState):
    return self.decoder.finish_generating(output, dec_state)

  def calc_nll(self, src: xnmt.Batch, trg: xnmt.Batch) -> xnmt.LossExpr:
    if isinstance(src, xnmt.structs.batch.CompoundBatch):
      src = src.batches[0]
    # Encode the sentence
    cur_losses = []
    dec_states = self.auto_regressive_states(src, trg)
    for i in range(trg.sent_len()):
      ref_word = xnmt.mark_as_batch([single_trg[i] for single_trg in trg])
      word_loss = self.decoder.calc_loss(dec_states[i], ref_word)
      if trg.mask is not None:
        word_loss = trg.mask.cmult_by_timestep_expr(word_loss, i, inverse=True)
      cur_losses.append(word_loss)
    return xnmt.LossExpr(dy.esum(cur_losses), [t.len_unpadded() for t in trg])

  def add_input(self, word, state: models.DecoderState) -> models.DecoderState:
    return self.decoder.add_input(state, word)

  def best_k(self, state: models.DecoderState, k: int, normalize_scores: bool = False):
    return self.decoder.best_k(state, k, normalize_scores)

  def sample(self, state: models.DecoderState, n: int, temperature: float = 1.0):
    return self.decoder.sample(state, n, temperature)

  def auto_regressive_states(self, src: xnmt.Batch, trg: xnmt.Batch):
    decoder_states = []
    prev_word, dec_state = None, None
    for i in range(trg.sent_len()):
      if prev_word is not None:
        dec_state = self.decoder.add_input(dec_state, prev_word)
      else:
        dec_state = self.initial_state(src)
      decoder_states.append(dec_state)
    return decoder_states

  def generate(self, src: xnmt.Batch, search_strategy: models.SearchStrategy, is_sort=True):
    outputs = []
    for i in range(src.batch_size()):
      src_i = xnmt.mark_as_batch(src[i])
      xnmt.event_trigger.start_sent(src_i)
      enc_result = self.encoder.encode(src_i)
      search_hyps = search_strategy.generate_output(self, self.decoder.initial_state(enc_result))
      for search_hyp in search_hyps:
        actions = search_hyp.actions()
        word_ids = [action.action_id[0] for action in actions]
        if hasattr(actions[0].decoder_state, "attender_state"):
          attentions = [action.decoder_state.attender_state.attention for action in actions]
        else:
          attentions = None
        sent = xnmt.structs.sentences.SimpleSentence(
             word_ids, src[i].idx, vocab=getattr(self.trg_reader, "vocab", None),
             output_procs=self.trg_reader.output_procs, score=search_hyp.score)
        if len(search_hyps) == 1:
          outputs.append(sent)
        else:
          outputs.append(xnmt.structs.sentences.NbestSentence(sent, src[i].idx))
    return outputs


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
