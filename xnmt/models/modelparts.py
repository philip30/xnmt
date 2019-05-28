import dynet as dy

from typing import List, Sequence, Tuple, Any, Union, Optional

import xnmt
import xnmt.models as models
import xnmt.models.states as states


class Attender(object):
  """
  A template class for functions implementing attention.
  """

  def initial_state(self, sent: Optional[xnmt.ExpressionSequence] = None) -> states.AttenderState:
    if sent is None:
      return states.AttenderState()
    else:
      return self.not_empty_initial_state(sent)

  def not_empty_initial_state(self, sent: xnmt.ExpressionSequence):
    """Args:
         sent: the encoder states, aka keys and values. Usually but not necessarily an :class:`expression_seqs.ExpressionSequence`
    """
    raise NotImplementedError('init_sent must be implemented for Attender subclasses')

  def calc_attention(self, decoder_context: dy.Expression, attender_state: states.AttenderState) -> dy.Expression:
    scores = self.calc_scores(decoder_context, attender_state)
    if attender_state.input_mask is not None:
      scores = attender_state.input_mask.add_to_tensor_expr(scores, multiplicator=-xnmt.globals.INF)
    if attender_state.read_mask is not None:
      scores = attender_state.read_mask.add_to_tensor_expr(scores, multiplicator=-xnmt.globals.INF)
    return dy.softmax(scores)

  def calc_scores(self, decoder_context: dy.Expression, attender_state: states.AttenderState) -> dy.Expression:
    """ Compute attention weights.

    Args:
    Returns:
      DyNet expression containing normalized attention scores
    """
    raise NotImplementedError('calc_attention must be implemented for Attender subclasses')

  def calc_context(self, decoder_context: dy.Expression, attender_state: states.AttenderState) -> Tuple[dy.Expression, states.AttenderState]:
    """ Compute weighted sum.

    Args:
    """
    attention = self.calc_attention(decoder_context, attender_state)
    context = attender_state.curr_sent * attention
    return context, states.AttenderState(
      curr_sent=attender_state.curr_sent, sent_context=attender_state.sent_context, input_mask=attender_state.input_mask,
      read_mask=attender_state.read_mask, attention=attention
    )


class Bridge(object):
  """
  Responsible for initializing the decoder LSTM, based on the final encoder state
  """
  def decoder_init(self, enc_final_states: Sequence[states.FinalTransducerState]) -> List[dy.Expression]:
    """
    Args:
      enc_final_states: list of final states for each encoder layer
    Returns:
      list of initial hidden and cell expressions for each layer. List indices 0..n-1 hold hidden states, n..2n-1 hold cell states.
    """
    raise NotImplementedError("decoder_init() must be implemented by Bridge subclasses")


class Decoder(object):
  """
  A template class to convert a prefix of previously generated words and
  a context vector into a probability distribution over possible next words.
  """
  def initial_state(self, enc_results: states.EncoderState, src: xnmt.Batch) -> states.UniDirectionalState:
    raise NotImplementedError('must be implemented by subclasses')

  def add_input(self, dec_state: states.UniDirectionalState, trg_word: Optional[xnmt.Batch]) -> states.UniDirectionalState:
    raise NotImplementedError('must be implemented by subclasses')

  def calc_loss(self, dec_state: states.UniDirectionalState, ref_action: xnmt.Batch):
    raise NotImplementedError('must be implemented by subclasses')

  def best_k(self, dec_state: states.UniDirectionalState, k: int, normalize_scores=False) -> List[states.SearchAction]:
    raise NotImplementedError('must be implemented by subclasses')

  def sample(self, dec_state: states.UniDirectionalState, n: int, temperature=1.0) -> List[states.SearchAction]:
    raise NotImplementedError('must be implemented by subclasses')

  def pick_oracle(self, oracle, dec_state: states.UniDirectionalState) -> List[states.SearchAction]:
    raise NotImplementedError('must be implemented by subclasses')

  def finish_generating(self, dec_output: Any, dec_state: states.UniDirectionalState) -> bool:
    raise NotImplementedError('must be implemented by subclasses')


class Embedder(object):
  """
  An embedder takes in word IDs and outputs continuous vectors.

  This can be done on a word-by-word basis, or over a sequence.
  """

  def embed(self, word: Any) -> dy.Expression:
    """Embed a single word.

    Args:
      word: This will generally be an integer word ID, but could also be something like a string. It could
            also be batched, in which case the input will be a :class:`xnmt.batcher.Batch` of integers or other things.

    Returns:
      Expression corresponding to the embedding of the word(s).
    """
    raise NotImplementedError('embed must be implemented in Embedder subclasses')

  def embed_sent(self, x: Any) -> xnmt.ExpressionSequence:
    """Embed a full sentence worth of words. By default, just do a for loop.

    Args:
      x: This will generally be a list of word IDs, but could also be a list of strings or some other format.
         It could also be batched, in which case it will be a (possibly masked) :class:`xnmt.batcher.Batch` object

    Returns:
      An expression sequence representing vectors of each word in the input.
    """
    # single mode
    raise NotImplementedError()


  def choose_vocab(self,
                   vocab: xnmt.Vocab,
                   yaml_path: xnmt.Path,
                   src_reader: models.InputReader,
                   trg_reader: models.InputReader) -> int:
    """Choose the vocab for the embedder basd on the passed arguments

    This is done in order of priority of vocab, model+yaml_path

    Args:
      vocab: If None, try to obtain from ``src_reader`` or ``trg_reader``, depending on the ``yaml_path``
      yaml_path: Path of this embedder in the component hierarchy. Automatically determined when deserializing the YAML model.
      src_reader: Model's src_reader, if exists and unambiguous.
      trg_reader: Model's trg_reader, if exists and unambiguous.

    Returns:
      chosen vocab
    """
    if vocab is not None:
      return len(vocab)
    elif "encoder" in yaml_path:
      if src_reader is None or getattr(src_reader, "vocab", None) is None:
        raise ValueError("Could not determine src_embedder's size. "
                         "Please set its vocab_size or vocab member explicitly, or specify the vocabulary of src_reader ahead of time.")
      return len(getattr(src_reader, "vocab", []))
    elif "decoder" in yaml_path or "output_projector" in yaml_path:
      if trg_reader is None or getattr(trg_reader, "vocab", None) is None:
        raise ValueError("Could not determine target embedder's size. "
                         "Please set its vocab_size or vocab member explicitly, or specify the vocabulary of trg_reader ahead of time.")
      return len(getattr(trg_reader, "vocab", []))
    else:
      raise ValueError("Attempted to determine vocab size of {} (path: {}), but path was not src_embedder, trg_embedder, or output_projector, so it could not determine what part of the model to use. Please set vocab_size or vocab explicitly.".format(self.__class__, yaml_path))

  def choose_vocab_size(self,
                        vocab_size: int,
                        vocab: xnmt.Vocab,
                        yaml_path: xnmt.Path,
                        src_reader: models.InputReader,
                        trg_reader: models.InputReader) -> int:
    """Choose the vocab size for the embedder based on the passed arguments

    This is done in order of priority of vocab_size, vocab, model+yaml_path

    Args:
      vocab_size : vocab size or None
      vocab: vocab or None
      yaml_path: Path of this embedder in the component hierarchy. Automatically determined when YAML-deserializing.
      src_reader: Model's src_reader, if exists and unambiguous.
      trg_reader: Model's trg_reader, if exists and unambiguous.

    Returns:
      chosen vocab size
    """
    if vocab_size is not None:
      return vocab_size
    elif vocab is not None:
      return len(vocab)
    elif "encoder" in yaml_path:
      if src_reader is None or getattr(src_reader, "vocab", None) is None:
        raise ValueError("Could not determine src_embedder's size. "
                         "Please set its vocab_size or vocab member explicitly, or specify the vocabulary of src_reader ahead of time.")
      return len(getattr(src_reader, "vocab", []))
    elif "decoder" in yaml_path or "output_projector" in yaml_path:
      if trg_reader is None or getattr(trg_reader, "vocab", None) is None:
        raise ValueError("Could not determine target embedder's size. "
                         "Please set its vocab_size or vocab member explicitly, or specify the vocabulary of trg_reader ahead of time.")
      return len(getattr(trg_reader, "vocab", []))
    else:
      raise ValueError(f"Attempted to determine vocab size of {self.__class__} (path: {yaml_path}), "
                       f"but path was not src_embedder, decoder.embedder, or output_projector, so it could not determine what part of the model to use. "
                       f"Please set vocab_size or vocab explicitly.")

  def is_batchable(self):
    raise NotImplementedError()


class Encoder(object):

  def encode(self, src: xnmt.Batch) -> states.EncoderState:
    raise NotImplementedError()


class SequenceComposer(object):

  def compose(self, embeds: Union[dy.Expression, List[dy.Expression]]) -> dy.Expression:
    raise NotImplementedError()


class SeqTransducer(object):
  """
  A class that transforms one sequence of vectors into another, using :class:`expression_seqs.ExpressionSequence` objects as inputs and outputs.
  """

  def transduce(self, seq) -> states.EncoderState:
    """
    Parameters should be :class:`expression_seqs.ExpressionSequence` objects wherever appropriate

    Args:
      seq: An expression sequence representing the input to the transduction

    Returns:
      result of transduction, an expression sequence
    """
    raise NotImplementedError("SeqTransducer.transduce() must be implemented by SeqTransducer sub-classes")


class UniDiSeqTransducer(SeqTransducer):
  def transduce(self, seq) -> states.EncoderState:
    raise NotImplementedError()

  def initial_state(self, init: Any=None) -> states.UniDirectionalState:
    raise NotImplementedError()

  def add_input(self, word: Any, previous_state: states.UniDirectionalState, mask: Optional[xnmt.Mask]) \
      -> states.UniDirectionalState:
    raise NotImplementedError()


class BidiSeqTransducer(SeqTransducer):
  def transduce(self, seq) -> states.EncoderState:
    raise NotImplementedError()


class Scorer(object):
  """
  A template class of things that take in a vector and produce a
  score over discrete output items.
  """

  def calc_scores(self, x: dy.Expression) -> dy.Expression:
    """
    Calculate the score of each discrete decision, where the higher
    the score is the better the model thinks a decision is. These
    often correspond to unnormalized log probabilities.

    Args:
      x: The vector used to make the prediction
    """
    raise NotImplementedError('calc_scores must be implemented by subclasses of Scorer')

  def best_k(self, x: dy.Expression, k: int, normalize_scores: bool = False) -> List[Tuple[int, dy.Expression]]:
    """
    Returns a list of the k items with the highest scores. The items may not be
    in sorted order.

    Args:
      x: The vector used to make the prediction
      k: Number of items to return
      normalize_scores: whether to normalize the scores
    """
    raise NotImplementedError('best_k must be implemented by subclasses of Scorer')

  def sample(self, x: dy.Expression, n: int, temperature: Optional[float] = 1.0) -> List[Tuple[int, dy.Expression]]:
    """
    Return samples from the scores that are treated as probability distributions.
    """
    raise NotImplementedError('sample must be implemented by subclasses of Scorer')

  def calc_probs(self, x: dy.Expression) -> dy.Expression:
    """
    Calculate the normalized probability of a decision.

    Args:
      x: The vector used to make the prediction
    """
    raise NotImplementedError('calc_prob must be implemented by subclasses of Scorer')

  def calc_log_probs(self, x: dy.Expression) -> dy.Expression:
    """
    Calculate the log probability of a decision

    log(calc_prob()) == calc_log_prob()

    Both functions exist because it might help save memory.

    Args:
      x: The vector used to make the prediction
    """
    raise NotImplementedError('calc_log_prob must be implemented by subclasses of Scorer')

  def calc_loss(self, x: dy.Expression, y: xnmt.Batch) -> dy.Expression:
    """
    Calculate the loss incurred by making a particular decision.

    Args:
      x: The vector used to make the prediction
      y: The correct label(s)
    """
    raise NotImplementedError('calc_loss must be implemented by subclasses of Scorer')

  def _choose_vocab_size(self,
                         vocab_size: Optional[int],
                         vocab: Optional[xnmt.Vocab],
                         trg_reader: Optional[models.templates.InputReader]) -> int:
    """Choose the vocab size for the embedder based on the passed arguments.

    This is done in order of priority of vocab_size, vocab, model

    Args:
      vocab_size: vocab size or None
      vocab: vocab or None
      trg_reader: Model's trg_reader, if exists and unambiguous.

    Returns:
      chosen vocab size
    """
    if vocab_size is not None:
      return vocab_size
    elif vocab is not None:
      return len(vocab)
    elif trg_reader is None or getattr(trg_reader, "vocab", None) is None:
      raise ValueError(
        "Could not determine scorer's's output size. "
        "Please set its vocab_size or vocab member explicitly, or specify the vocabulary of trg_reader ahead of time.")
    else:
      return len(trg_reader.vocab)

  def can_loss_be_derived_from_scores(self):
    raise NotImplementedError()


class Transform(object):
  """
  A class of transforms that change a dynet expression into another.
  """
  def transform(self, input_expr: dy.Expression) -> dy.Expression:
    raise NotImplementedError('transform() must be implemented in subclasses of Transform')


