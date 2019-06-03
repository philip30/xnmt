import numpy as np
import dynet as dy
import collections.abc as abc

from typing import List

import xnmt
import xnmt.models as models
import xnmt.modules.nn as nn

from typing import Optional


class ArbSeqLenUniDirectionalState(models.UniDirectionalState):
  """A state holding all the information needed for AutoRegressiveDecoder

  Args:
    rnn_state: a DyNet RNN state
    context: a DyNet expression
  """
  def __init__(self,
               rnn_state: models.UniDirectionalState,
               context: dy.Expression,
               attender_state: Optional[models.AttenderState],
               src: Optional[xnmt.Batch],
               prev_embedding: Optional[dy.Expression] = None):
    self._rnn_state = rnn_state
    self._attender_state = attender_state
    self._context = context
    self._src = src
    self._prev_embedding = prev_embedding

  @property
  def attender_state(self):
    return self._attender_state

  def add_input(self, word: dy.Expression, mask: Optional[xnmt.Mask] = None):
    raise NotImplementedError("Should not call this line of code")

  def context(self):
    return self._context

  @property
  def rnn_state(self):
    return self._rnn_state

  @property
  def src(self):
    return self._src

  def output(self):
    return self._rnn_state.output()
 
  @property
  def prev_embedding(self):
    return self._prev_embedding
  
  def position(self):
    return self._rnn_state.position()


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
               attender: Optional[models.Attender] = xnmt.bare(nn.MlpAttender),
               input_feeding: bool = True,
               bridge: models.Bridge = xnmt.bare(nn.CopyBridge),
               rnn: models.UniDiSeqTransducer = xnmt.bare(nn.UniLSTMSeqTransducer),
               transform: models.Transform = xnmt.bare(nn.AuxNonLinear),
               scorer: models.Scorer = xnmt.bare(nn.Softmax),
               init_with_bos: bool = True,
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
    self.init_with_bos = init_with_bos
    rnn_input_dim = embedder.emb_dim
    if input_feeding:
      rnn_input_dim += input_dim
    assert rnn_input_dim == rnn.total_input_dim, "Wrong input dimension in RNN layer: {} != {}".format(rnn_input_dim, rnn.total_input_dim)

  def initial_state(self, enc_results: models.EncoderState, src: xnmt.Batch) -> models.UniDirectionalState:
    """Get the initial state of the decoder given the encoder final states.

    Args:
      enc_results: The result of an encodings.
      src
    Returns:
      initial decoder state
    """
    attender_state = self.attender.initial_state(enc_results.encode_seq) if self.attender is not None else None
    batch_size = enc_results.encode_seq[0].dim()[1]
    rnn_state = self.rnn.initial_state(self.bridge.decoder_init(enc_results.encoder_final_states))
    zeros = dy.zeros(self.input_dim, batch_size=src.batch_size())
    if self.init_with_bos:
      ss_expr = self.embedder.embed(xnmt.mark_as_batch([xnmt.Vocab.SS] * batch_size))
      if self.input_feeding:
        ss_expr = dy.concatenate([ss_expr, zeros])
      rnn_state = rnn_state.add_input(ss_expr if self.input_feeding else ss_expr)
    return ArbSeqLenUniDirectionalState(rnn_state=rnn_state, context=zeros, attender_state=attender_state, src=src)

  def add_input(self, dec_state: ArbSeqLenUniDirectionalState, trg_word: xnmt.Batch,
                first_write: Optional[xnmt.Mask]=None) -> models.UniDirectionalState:
    """
    Add an input and return a *new* update the state.

    Args:
      dec_state: An object containing the current state.
      trg_word: The word to input.
      first_write
    Returns:
      The updated decoder state.
    """
    rnn_state = dec_state._rnn_state
    prev_context = dec_state.context()

    if trg_word is not None:
      trg_embedding = self.embedder.embed(trg_word, position=dec_state.position())
      inp_context = trg_embedding if not self.input_feeding else dy.concatenate([trg_embedding, prev_context])
      rnn_state = rnn_state.add_input(inp_context, trg_word.mask)
    else:
      trg_embedding = None
    # Calc Artention
    if self.attender is not None:
      context, attender_state = self.attender.calc_context(rnn_state.output(), dec_state.attender_state)
    else:
      context, attender_state = prev_context, None
    # Masking as needed
    if trg_word is not None and trg_word.mask is not None:
      ret_context = trg_word.mask.cmult_by_timestep_expr(context, 0, inverse=True) + \
                    trg_word.mask.cmult_by_timestep_expr(prev_context, 0, inverse=False)
      trg_embedding = trg_word.mask.cmult_by_timestep_expr(trg_embedding, 0, inverse=True)
      if dec_state.prev_embedding is not None:
        trg_embedding += trg_word.mask.cmult_by_timestep_expr(dec_state.prev_embedding, 0, inverse=False)
    else:
      ret_context = context
    if first_write is not None:
      ret_context += first_write.cmult_by_timestep_expr(context, 0, inverse=True)

    return ArbSeqLenUniDirectionalState(rnn_state=rnn_state, context=ret_context, attender_state=attender_state,
                                        src=dec_state.src, prev_embedding=trg_embedding)


  def _calc_transform(self, dec_state: ArbSeqLenUniDirectionalState) -> dy.Expression:
    h = dy.concatenate([dec_state.output(), dec_state.context()])
    return self.transform.transform(h)

  def best_k(self, dec_state: ArbSeqLenUniDirectionalState, k: int, normalize_scores: bool = False) \
      -> List[models.SearchAction]:
    h = self._calc_transform(dec_state)
    best_k = self.scorer.best_k(h, k, normalize_scores=normalize_scores)
    ret  = [models.SearchAction(dec_state, best_word, dy.pick(log_softmax, best_word), log_softmax, None) \
            for best_word, log_softmax in best_k]
    return ret

  def sample(self, dec_state: ArbSeqLenUniDirectionalState, n: int, temperature=1.0) \
      -> List[models.SearchAction]:
    h = self._calc_transform(dec_state)
    sample_k  = self.scorer.sample(h, n)
    ret  = [models.SearchAction(dec_state, best_word, dy.pick(log_softmax, best_word), log_softmax, None) \
            for best_word, log_softmax in sample_k]
    return ret

  def pick_oracle(self, oracle, dec_state: ArbSeqLenUniDirectionalState):
    log_prob = self.scorer.calc_log_probs(dec_state.output())
    return [models.SearchAction(dec_state, oracle, dy.pick_batch(log_prob, oracle), log_prob, None)]

  def calc_loss(self, dec_state: ArbSeqLenUniDirectionalState, ref_action: xnmt.Batch) -> dy.Expression:
    return self.scorer.calc_loss(self._calc_transform(dec_state), ref_action)

  def finish_generating(self, output, dec_state) -> bool:
    eog_symbol = self.eog_symbol
    if isinstance(output, abc.Sequence):
      return all([out_i == eog_symbol for out_i in output])
    else:
      return output == eog_symbol

