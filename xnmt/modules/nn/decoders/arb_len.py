import numpy as np
import dynet as dy

from typing import Any, List

import xnmt
import xnmt.models as models
import xnmt.modules.nn as nn
import xnmt.modules.nn.decoders.states as decoder_state


class ArbLenDecoder(models.Decoder, xnmt.Serializable):
  yaml_tag = "!ArbLenDecoder"
  """
  ArbSeqLenDecoder.

  Args:
    input_dim: input dimension
    embedder: embedder for target words
    input_feeding: whether to activate input feeding
    bridge: how to initialize decoder state
    rnn: recurrent decoder
    transform: a layer of transformation between rnn and output scorer
    scorer: the method of scoring the output (usually softmax)
  """
  @xnmt.serializable_init
  def __init__(self,
               input_dim: int = xnmt.default_layer_dim,
               embedder: models.Embedder = xnmt.bare(nn.LookupEmbedder),
               attender: models.Attender = xnmt.bare(nn.MlpAttender),
               input_feeding: bool = True,
               bridge: models.Bridge = xnmt.bare(nn.CopyBridge),
               rnn: models.UniDiSeqTransducer = xnmt.bare(nn.UniLSTMSeqTransducer),
               transform: models.Transform = xnmt.bare(nn.AuxNonLinear),
               scorer: models.Scorer = xnmt.bare(nn.Softmax),
               eog_symbol: int = xnmt.Vocab.ES) -> None:
    self.param_col = xnmt.param_manager(self)
    self.input_dim = input_dim
    self.embedder = embedder
    self.bridge = bridge
    self.rnn = rnn
    self.transform = transform
    self.scorer = scorer
    self.attender = attender
    # Input feeding
    self.input_feeding = input_feeding
    self.eog_symbol = eog_symbol
    rnn_input_dim = embedder.emb_dim
    if input_feeding:
      rnn_input_dim += input_dim
    assert rnn_input_dim == rnn.total_input_dim, "Wrong input dimension in RNN layer: {} != {}".format(rnn_input_dim, rnn.total_input_dim)

  def initial_state(self, enc_results: models.EncoderState) -> models.DecoderState:
    """Get the initial state of the decoder given the encoder final states.

    Args:
      enc_results: The result of an encodings.
    Returns:
      initial decoder state
    """
    attender_state = self.attender.initial_state(enc_results.encode_seq)
    batch_size = enc_results.encode_seq[0].dim()[1]
    rnn_state = self.rnn.initial_state(self.bridge.decoder_init(enc_results.encoder_final_states))
    zeros = dy.zeros(self.input_dim)
    ss_expr = self.embedder.embed(xnmt.mark_as_batch([xnmt.Vocab.SS] * batch_size))
    rnn_state = rnn_state.add_input(dy.concatenate([ss_expr, zeros]) if self.input_feeding else ss_expr)
    return decoder_state.ArbSeqLenDecoderState(rnn_state=rnn_state, context=zeros, attender_state=attender_state,
                                               timestep=0)

  def add_input(self, dec_state: decoder_state.ArbSeqLenDecoderState, trg_word: Any) -> models.DecoderState:
    """
    Add an input and return a *new* update the state.

    Args:
      dec_state: An object containing the current state.
      trg_word: The word to input.
    Returns:
      The updated decoder state.
    """
    rnn_state = dec_state.rnn_state
    if trg_word is not None:
      trg_embedding = self.embedder.embed(trg_word)
      context = trg_embedding if not self.input_feeding else dy.concatenate([trg_embedding, dec_state.context])
      rnn_state = rnn_state.add_input(context)
    context, attender_state = self.attender.calc_context(rnn_state.output(), dec_state.attender_state)
    return decoder_state.ArbSeqLenDecoderState(rnn_state=rnn_state, context=context, attender_state=attender_state,
                                               timestep=dec_state.timestep+1)


  def _calc_transform(self, dec_state: decoder_state.ArbSeqLenDecoderState) -> dy.Expression:
    h = dy.concatenate([dec_state.as_vector(), dec_state.context])
    return self.transform.transform(h)

  def best_k(self, dec_state: decoder_state.ArbSeqLenDecoderState, k: int, normalize_scores: bool = False) \
      -> List[models.SearchAction]:
    h = self._calc_transform(dec_state)
    best_k = self.scorer.best_k(h, k, normalize_scores=normalize_scores)
    ret  = [models.SearchAction(dec_state, best_word, best_score, None) for best_word, best_score in best_k]
    return ret

  def sample(self, dec_state: decoder_state.ArbSeqLenDecoderState, n: int, temperature=1.0) \
      -> List[models.SearchAction]:
    h = self._calc_transform(dec_state)
    sample_k  = self.scorer.sample(h, n)
    ret  = [models.SearchAction(dec_state, best_word, best_score, None) for best_word, best_score in sample_k]
    return ret

  def calc_loss(self, dec_state: decoder_state.ArbSeqLenDecoderState, ref_action: xnmt.Batch) -> dy.Expression:
    return self.scorer.calc_loss(self._calc_transform(dec_state), ref_action)

  def finish_generating(self, output, dec_state):
    eog_symbol = self.eog_symbol
    if type(output) == np.ndarray or type(output) == list:
      return [out_i == eog_symbol for out_i in output]
    else:
      return output == eog_symbol

