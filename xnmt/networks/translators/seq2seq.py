import dynet as dy

import xnmt.structs.batchers as batchers
import xnmt.modules.nn.encoders as encoders
import xnmt.inferences.search_output_processors as sp

from xnmt.modules.nn import decoders
from xnmt.internal.persistence import serializable_init, Serializable, bare
from xnmt.structs.losses import LossExpr

from xnmt.networks.translators import AutoRegressiveTranslator

class Seq2SeqTranslator(AutoRegressiveTranslator, Serializable):
  """
  A default translator based on attentional sequence-to-sequence networks.
  Args:
    encoder: An encoder to generate encoded inputs
    decoder: A decoder
  """

  yaml_tag = '!Seq2SeqTranslator'

  @serializable_init
  def __init__(self,
               encoder: encoders.Encoder = bare(encoders.Encoder),
               decoder: decoders.Decoder = bare(decoders.AutoRegressiveDecoder),
               search_output_processor = bare(sp.TranslatorProcessor)):
    super().__init__(search_output_processor)
    self.encoder = encoder
    self.decoder = decoder

  def initial_state(self, src: batchers.Batch) -> decoders.DecoderState:
    return self.decoder.initial_state(self.encoder.encode(src))

  def finish_generating(self, output, dec_state: decoders.DecoderState):
    return self.decoder.finish_generating(output, dec_state)

  def calc_nll(self, src: batchers.Batch, trg: batchers.Batch) -> LossExpr:
    if isinstance(src, batchers.CompoundBatch):
      src = src.batches[0]
    # Encode the sentence
    cur_losses = []
    dec_states = self.auto_regressive_states(src, trg)
    for i in range(trg.sent_len()):
      ref_word = batchers.mark_as_batch([single_trg[i] for single_trg in trg])
      word_loss = self.decoder.calc_loss(dec_states[i], ref_word)
      if trg.mask is not None:
        word_loss = trg.mask.cmult_by_timestep_expr(word_loss, i, inverse=True)
      cur_losses.append(word_loss)
    return LossExpr(dy.esum(cur_losses), [t.len_unpadded() for t in trg])

  def add_input(self, word, state: decoders.DecoderState) -> decoders.DecoderState:
    return self.decoder.add_input(state, word)

  def best_k(self, state: decoders.DecoderState, k: int, normalize_scores: bool = False):
    return self.decoder.best_k(state, k, normalize_scores)

  def sample(self, state: decoders.DecoderState, n: int, temperature: float = 1.0):
    return self.decoder.sample(state, n, temperature)

  def auto_regressive_states(self, src: batchers.Batch, trg: batchers.Batch):
    decoder_states = []
    prev_word, dec_state = None, None
    for i in range(trg.sent_len()):
      if prev_word is not None:
        dec_state = self.decoder.add_input(dec_state, prev_word)
      else:
        dec_state = self.initial_state(src)
      decoder_states.append(dec_state)
    return decoder_states
