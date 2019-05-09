import numbers
import functools
import collections

from typing import Any, Optional, Union

import numpy as np
import dynet as dy

import xnmt.param_initializers as pinit
import xnmt.sent as sent
import xnmt.vocabs as vocabs
import xnmt.input_readers as input_readers
import xnmt.batchers as batchers
import xnmt.events as events
import xnmt.expression_seqs as expression_seqs
import xnmt.modelparts.transforms as transforms
import xnmt.param_collections as param_collections
import xnmt.seq_composer as seq_composer

from xnmt import logger
from xnmt.persistence import bare, Path, Ref, Serializable, serializable_init


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

  def embed_sent(self, x: Any) -> expression_seqs.ExpressionSequence:
    """Embed a full sentence worth of words. By default, just do a for loop.

    Args:
      x: This will generally be a list of word IDs, but could also be a list of strings or some other format.
         It could also be batched, in which case it will be a (possibly masked) :class:`xnmt.batcher.Batch` object

    Returns:
      An expression sequence representing vectors of each word in the input.
    """
    # single mode
    if not batchers.is_batched(x):
      expr = expression_seqs.ExpressionSequence(expr_list=[self.embed(word) for word in x])
    # minibatch mode
    elif type(self) == LookupEmbedder:
      embeddings = []
      for word_i in range(x.sent_len()):
        batch = batchers.mark_as_batch([single_sent[word_i] for single_sent in x])
        embeddings.append(self.embed(batch))
      expr = expression_seqs.ExpressionSequence(expr_list=embeddings, mask=x.mask)
    else:
      assert type(x[0]) == sent.SegmentedSentence, "Need to use CharFromWordTextReader for non standard embeddings."
      embeddings = []
      all_embeddings = []
      for sentence in x:
        embedding = []
        for i in range(sentence.len_unpadded()):
          embed_word = self.embed(sentence.words[i])
          embedding.append(embed_word)
          all_embeddings.append(embed_word)
        embeddings.append(embedding)
      # Useful when using dy.autobatch
      dy.forward(all_embeddings)
      all_embeddings.clear()
      # Pad the results
      expr = batchers.pad_embedding(embeddings)

    return expr


  def choose_vocab(self,
                   vocab: vocabs.Vocab,
                   yaml_path: Path,
                   src_reader: input_readers.InputReader,
                   trg_reader: input_readers.InputReader) -> vocabs.Vocab:
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
    elif "src_embedder" in yaml_path:
      if src_reader is None or src_reader.vocab is None:
        raise ValueError("Could not determine src_embedder's vocabulary. Please set its vocab member explicitly, or specify the vocabulary of src_reader ahead of time.")
      return len(src_reader.vocab)
    elif "embedder" in yaml_path or "output_projector" in yaml_path:
      if trg_reader is None or trg_reader.vocab is None:
        raise ValueError("Could not determine trg_embedder's vocabulary. Please set its vocab member explicitly, or specify the vocabulary of trg_reader ahead of time.")
      return len(trg_reader.vocab)
    else:
      raise ValueError("Attempted to determine vocab size of {} (path: {}), but path was not src_embedder, trg_embedder, or output_projector, so it could not determine what part of the model to use. Please set vocab_size or vocab explicitly.".format(self.__class__, yaml_path))

  def choose_vocab_size(self,
                        vocab_size: numbers.Integral,
                        vocab: vocabs.Vocab,
                        yaml_path: Path,
                        src_reader: input_readers.InputReader,
                        trg_reader: input_readers.InputReader) -> int:
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
    elif "src_embedder" in yaml_path:
      if src_reader is None or getattr(src_reader,"vocab",None) is None:
        raise ValueError("Could not determine src_embedder's size. "
                         "Please set its vocab_size or vocab member explicitly, or specify the vocabulary of src_reader ahead of time.")
      return len(src_reader.vocab)
    elif "embedder" in yaml_path or "output_projector" in yaml_path:
      if trg_reader is None or trg_reader.vocab is None:
        raise ValueError("Could not determine target embedder's size. "
                         "Please set its vocab_size or vocab member explicitly, or specify the vocabulary of trg_reader ahead of time.")
      return len(trg_reader.vocab)
    else:
      raise ValueError(f"Attempted to determine vocab size of {self.__class__} (path: {yaml_path}), "
                       f"but path was not src_embedder, decoder.embedder, or output_projector, so it could not determine what part of the model to use. "
                       f"Please set vocab_size or vocab explicitly.")


class WordEmbedder(Embedder):
  """
  Word embeddings via full matrix.

  Args:
    emb_dim: embedding dimension
    weight_noise: apply Gaussian noise with given standard deviation to embeddings
    fix_norm: fix the norm of word vectors to be radius r, see https://arxiv.org/abs/1710.01329
  """

  @events.register_xnmt_handler
  def __init__(self,
               emb_dim: int,
               weight_noise: float,
               fix_norm: Optional[float] = None):
    self.fix_norm = fix_norm
    self.weight_noise = weight_noise
    self.emb_dim = emb_dim
    self.train = True

  @events.handle_xnmt_event
  def on_set_train(self, val: bool) -> None:
    self.train = val

  def embed(self, x: Union[batchers.Batch, numbers.Integral]) -> dy.Expression:
    """
    Embed a single word in a sentence.
    :param x: A word id.
    :return: Embedded word.
    """
    ret = self._embed_word(x, batchers.is_batched(x))
    ## Applying Fix normalization
    if self.fix_norm is not None:
      ret = dy.cdiv(ret, dy.l2_norm(ret)) * self.fix_norm
    ## Weight noise only when training
    if self.train and self.weight_noise > 0.0:
      ret = dy.noise(ret, self.weight_noise)
    return ret

  def _embed_word(self, word, is_batched):
    raise NotImplementedError()


class LookupEmbedder(WordEmbedder, transforms.Linear, Serializable):

  yaml_tag = '!LookupEmbedder'

  @serializable_init
  def __init__(self,
               emb_dim: int = Ref("exp_global.default_layer_dim"),
               vocab_size: Optional[int] = None,
               vocab: Optional[vocabs.Vocab] = None,
               yaml_path: Path = Path(''),
               src_reader: Optional[input_readers.InputReader] = Ref("model.src_reader", default=None),
               trg_reader: Optional[input_readers.InputReader] = Ref("model.trg_reader", default=None),
               is_dense: bool = False,
               param_init: pinit.ParamInitializer = Ref("exp_global.param_init", default=bare(pinit.GlorotInitializer)),
               bias_init: pinit.ParamInitializer = Ref("exp_global.bias_init", default=bare(pinit.ZeroInitializer)),
               init_fastext: Optional[str] = None,
               weight_noise: float = Ref("exp_global.weight_noise", default=0.0),
               fix_norm: Optional[float] = None):
    super().__init__(emb_dim=emb_dim, weight_noise=weight_noise, fix_norm=fix_norm)
    # Embedding Parameters
    pcol = param_collections.ParamManager.my_params(self)
    self.vocab_size = self.choose_vocab_size(vocab_size, vocab, yaml_path, src_reader, trg_reader)
    emb_mtr_dim = (self.vocab_size, self.emb_dim)

    if init_fastext is not None:
      logger.info("Setting Dense to False because of init_fastext")
      is_dense = False

    if not is_dense:
      if init_fastext is not None:
        self.embeddings = pcol.lookup_parameters_from_numpy(self._read_fasttext_embeddings(vocab, init_fastext))
      else:
        self.embeddings = pcol.add_lookup_parameters(emb_mtr_dim, init=param_init.initializer(emb_mtr_dim,  is_lookup=True))
    else:
      self.embeddings = pcol.add_parameters(emb_mtr_dim, init=param_init.initializer(emb_mtr_dim, is_lookup=True))
      self.bias = pcol.add_parameters((self.vocab_size,), init=bias_init.initializer((self.vocab_size,)))

    # Model States
    self.is_dense = is_dense
    self.train = False
    self.save_processed_arg("vocab_size", self.vocab_size)

  def _embed_word(self, word, is_batched):
    if is_batched:
      embedding = dy.pick_batch(self.embeddings, word) if self.is_dense else self.embeddings.batch(word)
    else:
      embedding = dy.pick(self.embeddings, index=word) if self.is_dense else self.embeddings[word]
    return embedding

  def transform(self, input_expr: dy.Expression) -> dy.Expression:
    if self.is_dense:
      w = dy.parameter(self.embeddings)
      b = dy.parameter(self.bias)
    else:
      raise NotImplementedError("Non dense embedder transform is not implemented yet.")

    return dy.affine_transform([b, w, input_expr])

  def _read_fasttext_embeddings(self, vocab: vocabs.Vocab, init_fastext):
    """
    Reads FastText embeddings from a file. Also prints stats about the loaded embeddings for sanity checking.

    Args:
      vocab: a `Vocab` object containing the vocabulary for the experiment
      embeddings_file_handle: A file handle on the embeddings file. The embeddings must be in FastText text
                              format.
    Returns:
      tuple: A tuple of (total number of embeddings read, # embeddings that match vocabulary words, # vocabulary words
     without a matching embedding, embeddings array).
    """
    with open(init_fastext, encoding='utf-8') as embeddings_file_handle:
      _, dimension = next(embeddings_file_handle).split()
      if int(dimension) != self.emb_dim:
        raise Exception(f"An embedding size of {self.emb_dim} was specified, but the pretrained embeddings have size {dimension}")

      # Poor man's Glorot initializer for missing embeddings
      bound = np.sqrt(6/(self.vocab_size + self.emb_dim))

      total_embs = 0
      in_vocab = 0
      missing = 0

      embeddings = np.empty((self.vocab_size, self.emb_dim), dtype='float')
      found = np.zeros(self.vocab_size, dtype='bool_')

      for line in embeddings_file_handle:
        total_embs += 1
        word, vals = line.strip().split(' ', 1)
        if word in vocab.w2i:
          in_vocab += 1
          index = vocab.w2i[word]
          embeddings[index] = np.fromstring(vals, sep=" ")
          found[index] = True

      for i in range(self.vocab_size):
        if not found[i]:
          missing += 1
          embeddings[i] = np.random.uniform(-bound, bound, self.emb_dim)

      logger.info(f"{in_vocab} vocabulary matches out of {total_embs} total embeddings; "
                  f"{missing} vocabulary words without a pretrained embedding out of {self.vocab_size}")

    return embeddings


class BagOfWordsEmbedder(WordEmbedder, Serializable):

  yaml_tag = '!BagOfWordsEmbedder'
  ONE_MB = 1000 * 1024

  @serializable_init
  def __init__(self,
               emb_dim = Ref("exp_global.default_layer_dim"),
               ngram_vocab: vocabs.Vocab = None,
               word_vocab: Optional[vocabs.Vocab] = Ref("model.src_reader.vocab", default=None),
               char_vocab: Optional[vocabs.Vocab] = Ref("model.src_reader.char_vocab", default=None),
               ngram_size: int = 1,
               transform: Optional[transforms.Transform] = None,
               include_lower_ngrams: bool = True,
               weight_noise: float = Ref("exp_global.weight_noise", default=0.0),
               fix_norm: Optional[float] = None):
    super().__init__(emb_dim=emb_dim, weight_noise=weight_noise, fix_norm=fix_norm)
    self.transform = self.add_serializable_component("transform", transform,
                                                     lambda: transforms.NonLinear(input_dim=ngram_vocab.vocab_size(),
                                                                                  output_dim=emb_dim,
                                                                                  activation='relu'))
    self.word_vocab = word_vocab
    self.char_vocab = char_vocab
    self.ngram_vocab = ngram_vocab
    self.ngram_size = ngram_size
    self.include_lower_ngrams = include_lower_ngrams

  @functools.lru_cache(maxsize=ONE_MB)
  def to_ngram_stats(self, word):
    word_vector = collections.defaultdict(int)

    if self.word_vocab is not None:
      chars = self.word_vocab[word]
    elif self.char_vocab is not None:
      chars = "".join([self.char_vocab[c] for c in word if c != self.char_vocab.PAD and c != self.char_vocab.SS])
    else:
      raise ValueError("Either word vocab or char vocab should not be None")

    # This offset is used to generate bag-of-words for a specific ngrams only
    # For example 3-grams which is used in some papers
    offset = self.ngram_size-1 if not self.include_lower_ngrams else 0

    # Fill in word_vecs
    for i in range(len(chars)):
      for j in range(i+offset, min(i+self.ngram_size, len(chars))):
        word_vector[chars[i:j+1]] += 1

    return word_vector

  def _embed_word(self, segmented_word, is_batched):
    if self.word_vocab is not None:
      ngram_stats = self.to_ngram_stats(segmented_word.word)
    elif self.char_vocab is not None:
      ngram_stats = self.to_ngram_stats(segmented_word.chars)
    else:
      raise ValueError("Either word vocab or char vocab should not be None")

    not_in = [key for key in ngram_stats.keys() if key not in self.ngram_vocab.w2i]
    for key in not_in:
      ngram_stats.pop(key)

    if len(ngram_stats) > 0:
      ngrams = [self.ngram_vocab.convert(ngram) for ngram in ngram_stats.keys()]
      counts = list(ngram_stats.values())
    else:
      ngrams = [self.ngram_vocab.UNK]
      counts = [1]

    input_tensor = dy.sparse_inputTensor([ngrams], counts, (self.ngram_vocab.vocab_size(),))
    # Note: If one wants to use CHARAGRAM embeddings, use NonLinear with Relu.
    return self.transform.transform(input_tensor)


class CharCompositionEmbedder(WordEmbedder, Serializable):

  yaml_tag = '!CharCompositionEmbedder'

  @serializable_init
  def __init__(self,
               char_vocab: Optional[vocabs.CharVocab] = Ref("model.src_reader.char_vocab", default=None),
               vocab_size: Optional[int] = None,
               emb_dim: int = Ref("exp_global.default_layer_dim"),
               weight_noise: float = Ref("exp_global.weight_noise", default=0.0),
               param_init: pinit.ParamInitializer = Ref("exp_global.param_init", default=bare(pinit.GlorotInitializer)),
               bias_init: pinit.ParamInitializer = Ref("exp_global.bias_init", default=bare(pinit.ZeroInitializer)),
               composer: seq_composer.SequenceComposer = bare(seq_composer.SumComposer),
               fix_norm: Optional[float] = None):
    super().__init__(emb_dim=emb_dim, weight_noise=weight_noise, fix_norm=fix_norm)
    self.composer = composer
    # Embedding Parameters
    pcol = param_collections.ParamManager.my_params(self)
    self.vocab_size = self.choose_vocab_size(vocab_size, char_vocab, '', None, None)
    self.embeddings = pcol.add_lookup_parameters((self.vocab_size, self.emb_dim), init=param_init.initializer((self.vocab_size, self.emb_dim),  is_lookup=True))
    # Model States
    self.train = False
    self.save_processed_arg("vocab_size", self.vocab_size)


  def _embed_word(self, word: sent.SegmentedWord, is_batched: bool = False):
    char_embeds = self.embeddings.batch(batchers.mark_as_batch(word.chars))

    char_embeds = [dy.pick_batch_elem(char_embeds, i) for i in range(len(word.chars))]
    return self.composer.compose(char_embeds)


class CompositeEmbedder(Embedder, Serializable):
  yaml_tag = '!CompositeEmbedder'

  @serializable_init
  def __init__(self, embedders):
    self.embedders = embedders

  def embed_sent(self, x: Any):
    embeddings = [embedder.embed_sent(x) for embedder in self.embedders]
    ret = []
    for j in range(len(embeddings[0])):
      ret.append(dy.esum([embeddings[i][j] for i in range(len(embeddings))]))
    return expression_seqs.ExpressionSequence(expr_list=ret, mask=embeddings[0].mask)

  def embed(self, word: Any) -> dy.Expression:
    def select_word(_word, _embedder):
      if type(_word) == sent.SegmentedWord and type(_embedder) == LookupEmbedder:
        _word = _word.word
      return _word

    return dy.esum([embedder.embed(select_word(word, embedder)) for embedder in self.embedders])


class NoopEmbedder(Embedder, Serializable):
  """
  This embedder performs no lookups but only passes through the inputs.

  Normally, the input is a Sentence object, which is converted to an expression.

  Args:
    emb_dim: Size of the inputs
  """

  yaml_tag = '!NoopEmbedder'

  @serializable_init
  def __init__(self, emb_dim: Optional[numbers.Integral]) -> None:
    self.emb_dim = emb_dim

  def embed(self, x: Union[np.ndarray, list]) -> dy.Expression:
    return dy.inputTensor(x, batched=batchers.is_batched(x))

  def embed_sent(self, x: sent.Sentence) -> expression_seqs.ExpressionSequence:
    # TODO refactor: seems a bit too many special cases that need to be distinguished
    batched = batchers.is_batched(x)
    first_sent = x[0] if batched else x
    if hasattr(first_sent, "get_array"):
      if not batched:
        return expression_seqs.LazyNumpyExpressionSequence(lazy_data=x.get_array())
      else:
        return expression_seqs.LazyNumpyExpressionSequence(lazy_data=batchers.mark_as_batch([s for s in x]), mask=x.mask)
    else:
      if not batched:
        embeddings = [self.embed(word) for word in x]
      else:
        embeddings = []
        for word_i in range(x.sent_len()):
          embeddings.append(self.embed(batchers.mark_as_batch([single_sent[word_i] for single_sent in x])))
      return expression_seqs.ExpressionSequence(expr_list=embeddings, mask=x.mask)


class PositionEmbedder(Embedder, Serializable):

  yaml_tag = '!PositionEmbedder'

  @serializable_init
  def __init__(self,
               max_pos: numbers.Integral,
               emb_dim: numbers.Integral = Ref("exp_global.default_layer_dim"),
               param_init: pinit.ParamInitializer = Ref("exp_global.param_init", default=bare(pinit.GlorotInitializer))):
    """
    max_pos: largest embedded position
    emb_dim: embedding size
    param_init: how to initialize embedding matrix
    """
    self.max_pos = max_pos
    self.emb_dim = emb_dim
    param_collection = param_collections.ParamManager.my_params(self)
    dim = (self.emb_dim, max_pos)
    self.embeddings = param_collection.add_parameters(dim, init=param_init.initializer(dim, is_lookup=True))

  def embed(self, word): raise NotImplementedError("Position-embedding for individual words not implemented yet.")
  def embed_sent(self, sent_len: numbers.Integral) -> expression_seqs.ExpressionSequence:
    embeddings = dy.strided_select(dy.parameter(self.embeddings), [1,1], [0,0], [self.emb_dim, sent_len])
    return expression_seqs.ExpressionSequence(expr_tensor=embeddings, mask=None)


