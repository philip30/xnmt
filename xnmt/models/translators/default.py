import dynet as dy
import numpy as np

import numbers
from typing import Any, List, Sequence, Union

import xnmt.batchers as batchers
import xnmt.event_trigger as event_trigger
import xnmt.events as events
import xnmt.inferences as inferences
import xnmt.input_readers as input_readers
import xnmt.search_strategies as search_strategies
import xnmt.sent as sent
import xnmt.vocabs as vocabs

from xnmt.settings import settings
from xnmt.modelparts import attenders, decoders, embedders
from xnmt.transducers import recurrent, base as transducers_base
from xnmt.persistence import serializable_init, Serializable, bare
from xnmt.reports import Reportable
from xnmt.losses import LossExpr

from .auto_regressive import AutoRegressiveTranslator


class DefaultTranslator(AutoRegressiveTranslator, Serializable, Reportable):
  """
  A default translator based on attentional sequence-to-sequence models.
  Args:
    src_reader: A reader for the source side.
    trg_reader: A reader for the target side.
    src_embedder: A word embedder for the input language
    encoder: An encoder to generate encoded inputs
    attender: An attention module
    decoder: A decoder
    inference: The default inference strategy used for this model
  """

  yaml_tag = '!DefaultTranslator'

  @events.register_xnmt_handler
  @serializable_init
  def __init__(self,
               src_reader: input_readers.InputReader,
               trg_reader: input_readers.InputReader,
               src_embedder: embedders.Embedder = bare(embedders.LookupEmbedder),
               encoder: transducers_base.SeqTransducer = bare(recurrent.BiLSTMSeqTransducer),
               attender: attenders.Attender = bare(attenders.MlpAttender),
               decoder: decoders.Decoder = bare(decoders.AutoRegressiveDecoder),
               inference: inferences.AutoRegressiveInference = bare(inferences.AutoRegressiveInference),
               truncate_dec_batches: bool = False) -> None:
    super().__init__(src_reader=src_reader, trg_reader=trg_reader)
    self.src_embedder = src_embedder
    self.encoder = encoder
    self.attender = attender
    self.decoder = decoder
    self.inference = inference
    self.truncate_dec_batches = truncate_dec_batches

  def shared_params(self):
    return [{".src_embedder.emb_dim", ".encoder.input_dim"},
            {".encoder.hidden_dim", ".attender.input_dim", ".decoder.input_dim"},
            {".attender.state_dim", ".decoder.rnn.hidden_dim"}]

  def _initial_state(self, src: Union[batchers.Batch, sent.Sentence]):
    # Encode sentence and initiate decoder state
    embeddings = self.src_embedder.embed_sent(src)
    encoding = self.encoder.transduce(embeddings)
    final_state = self.encoder.get_final_states()
    self.attender.init_sent(encoding)
    self.decoder.init_sent(encoding)
    ss = batchers.mark_as_batch([vocabs.Vocab.SS] * src.batch_size()) if batchers.is_batched(src) else vocabs.Vocab.SS
    initial_state = self.decoder.initial_state(final_state, ss)
    return initial_state

  def eog_symbol(self):
    return self.decoder.eog_symbol()

  def finish_generating(self, output, dec_state):
    return self.decoder.finish_generating(output, dec_state)

  def calc_nll(self,
               src: Union[batchers.Batch, sent.Sentence],
               trg: Union[batchers.Batch, sent.Sentence]) -> LossExpr:
    if isinstance(src, batchers.CompoundBatch):
      src = src.batches[0]
    # Encode the sentence
    initial_state = self._initial_state(src)

    dec_state = initial_state
    trg_mask = trg.mask if batchers.is_batched(trg) else None
    cur_losses = []
    seq_len = trg.sent_len()

    # Sanity check if requested
    if settings.CHECK_VALIDITY and batchers.is_batched(src):
      for j, single_trg in enumerate(trg):
        # assert consistent length
        assert single_trg.sent_len() == seq_len
        # assert exactly one unmasked ES token
        assert 1 == len([i for i in range(seq_len) if (trg_mask is None or
                                                       trg_mask.np_arr[j,i]==0) and single_trg[i]==vocabs.Vocab.ES])

    input_word = None
    for i in range(seq_len):
      ref_word = self._select_ref_words(trg, i, truncate_masked=self.truncate_dec_batches)

      if input_word is not None:
        dec_state = self.decoder.add_input(dec_state, input_word)
      rnn_output = dec_state.as_vector()
      dec_state.context = self.attender.calc_context(rnn_output)
      word_loss = self.decoder.calc_loss(dec_state, ref_word)

      if not self.truncate_dec_batches and batchers.is_batched(src) and trg_mask is not None:
        word_loss = trg_mask.cmult_by_timestep_expr(word_loss, i, inverse=True)
      cur_losses.append(word_loss)
      input_word = ref_word
    units = [t.len_unpadded() for t in trg]
    return LossExpr(dy.esum(cur_losses), units)

  def _select_ref_words(self, sentence, index, truncate_masked = False):
    if truncate_masked:
      mask = sentence.mask if batchers.is_batched(sentence) else None
      if not batchers.is_batched(sentence):
        return sentence[index]
      else:
        ret = []
        found_masked = False
        for (j, single_trg) in enumerate(sentence):
          if mask is None or mask.np_arr[j, index] == 0 or np.sum(mask.np_arr[:, index]) == mask.np_arr.shape[0]:
            assert not found_masked, "sentences must be sorted by decreasing target length"
            ret.append(single_trg[index])
          else:
            found_masked = True
        return batchers.mark_as_batch(ret)
    else:
      return sentence[index] if not batchers.is_batched(sentence) else \
             batchers.mark_as_batch([single_trg[index] for single_trg in sentence])

  def generate_search_output(self,
                             src: batchers.Batch,
                             search_strategy: search_strategies.SearchStrategy) -> List[search_strategies.SearchOutput]:
    """
    Takes in a batch of source sentences and outputs a list of search outputs.
    Args:
      src: The source sentences
      search_strategy: The strategy with which to perform the search
    Returns:
      A list of search outputs including scores, etc.
    """
    if src.batch_size() != 1:
      raise NotImplementedError("batched decoding not implemented for DefaultTranslator. "
                                "Specify inference batcher with batch size 1.")
    if isinstance(src, batchers.CompoundBatch):
      src = src.batches[0]
    search_outputs = search_strategy.generate_output(self,
                                                     self._initial_state(src),
                                                     src_length=src.sent_len())
    return search_outputs

  def generate(self,
               src: batchers.Batch,
               search_strategy: search_strategies.SearchStrategy) -> Sequence[sent.Sentence]:
    """
    Takes in a batch of source sentences and outputs a list of search outputs.
    Args:
      src: The source sentences
      search_strategy: The strategy with which to perform the search
    Returns:
      A list of search outputs including scores, etc.
    """
    assert src.batch_size() == 1
    event_trigger.start_sent(src)
    search_outputs = self.generate_search_output(src, search_strategy)
    if isinstance(src, batchers.CompoundBatch): src = src.batches[0]
    sorted_outputs = sorted(search_outputs, key=lambda x: x.score[0], reverse=True)
    assert len(sorted_outputs) >= 1
    outputs = []
    for curr_output in sorted_outputs:
      output_actions = [x for x in curr_output.word_ids[0]]
      attentions = [x for x in curr_output.attentions[0]]
      score = curr_output.score[0]
      out_sent = self._emit_translation(src, output_actions, score)
      if len(sorted_outputs) == 1:
        outputs.append(out_sent)
      else:
        outputs.append(sent.NbestSentence(base_sent=out_sent, nbest_id=src[0].idx))

    if self.is_reporting():
      attentions = np.concatenate([x.npvalue() for x in attentions], axis=1)
      self.report_sent_info({"attentions": attentions,
                             "src": src[0],
                             "output": outputs[0]})

    return outputs

  def _emit_translation(self, src, output_actions, score):
    return sent.SimpleSentence(idx=src[0].idx,
                               words=output_actions,
                               vocab=getattr(self.trg_reader, "vocab", None),
                               output_procs=self.trg_reader.output_procs,
                               score=score)

  def add_input(self, word: Any, state: decoders.AutoRegressiveDecoderState) -> AutoRegressiveTranslator.Output:
    if word is not None:
      if type(word) == int:
        word = [word]
      if type(word) == list or type(word) == np.ndarray:
        word = batchers.mark_as_batch(word)
    next_state = self.decoder.add_input(state, word) if word is not None else state
    attention = self.attender.calc_attention(next_state.as_vector())
    next_state.context = self.attender.calc_context(next_state.as_vector(), attention=attention)
    return AutoRegressiveTranslator.Output(next_state, attention)

  def best_k(self, state: decoders.AutoRegressiveDecoderState, k: numbers.Integral, normalize_scores: bool = False):
    best_words, best_scores = self.decoder.best_k(state, k, normalize_scores)
    return best_words, best_scores

  def sample(self, state: decoders.AutoRegressiveDecoderState, n: numbers.Integral, temperature: float = 1.0):
    return self.decoder.sample(state, n, temperature)


class TreeTranslator(DefaultTranslator, Serializable):
  yaml_tag = "!TreeTranslator"

  def _emit_translation(self, src, output_actions, score):
    return sent.DepTreeRNNGSequenceSentence(idx=src[0].idx,
                                            score=score,
                                            actions=output_actions,
                                            surface_vocab=getattr(self.trg_reader, "surface_vocab", None),
                                            nt_vocab=getattr(self.trg_reader, "nt_vocab", None),
                                            edge_vocab=getattr(self.trg_reader, "edge_vocab", None),
                                            output_procs=self.trg_reader.output_procs)


