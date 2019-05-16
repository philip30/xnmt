import functools
import collections
import numpy as np
import dynet as dy
from typing import Any, Optional, Union

import xnmt
import xnmt.models as models
import xnmt.structs.sentences as sent
import xnmt.modules.nn.transforms as transforms
import xnmt.modules.nn.composers as composers


class WordEmbedder(models.Embedder):
  """
  Word embeddings via full matrix.

  Args:
    emb_dim: embedding dimension
    weight_noise: apply Gaussian noise with given standard deviation to embeddings
    fix_norm: fix the norm of word vectors to be radius r, see https://arxiv.org/abs/1710.01329
  """
  def __init__(self,
               emb_dim: int,
               weight_noise: float,
               fix_norm: Optional[float] = None):
    self.fix_norm = fix_norm
    self.weight_noise = weight_noise
    self.emb_dim = emb_dim
    self.train = True

  def embed(self, x: Union[xnmt.Batch, int]) -> dy.Expression:
    """
    Embed a single word in a sentence.
    :param x: A word id.
    :return: Embedded word.
    """
    ret = self._embed_word(x, xnmt.is_batched(x))
    ## Applying Fix normalization
    if self.fix_norm is not None:
      ret = dy.cdiv(ret, dy.l2_norm(ret)) * self.fix_norm
    ## Weight noise only when training
    if self.train and self.weight_noise > 0.0:
      ret = dy.noise(ret, self.weight_noise)
    return ret

  def embed_sent(self, x: Any):
    if not xnmt.is_batched(x):
      expr = xnmt.ExpressionSequence(expr_list=[self.embed(word) for word in x])
    # minibatch mode
    elif self.is_batchable():
      embeddings = []
      for word_i in range(x.sent_len()):
        batch = xnmt.mark_as_batch([single_sent[word_i] for single_sent in x])
        embeddings.append(self.embed(batch))
      expr = xnmt.ExpressionSequence(expr_list=embeddings, mask=x.mask)
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
      expr = xnmt.structs.batchers.pad_embedding(embeddings)

    return expr

  def _embed_word(self, word, is_batched):
    raise NotImplementedError()

  def is_batchable(self):
    raise NotImplementedError()


class LookupEmbedder(WordEmbedder, transforms.Linear, xnmt.Serializable):
  @xnmt.serializable_init
  def __init__(self,
               emb_dim: int = xnmt.default_layer_dim,
               vocab_size: Optional[int] = None,
               vocab: Optional[xnmt.Vocab] = None,
               yaml_path: xnmt.Path = xnmt.Path(''),
               src_reader: Optional[models.InputReader] = xnmt.ref_src_reader,
               trg_reader: Optional[models.InputReader] = xnmt.ref_trg_reader,
               is_dense: bool = False,
               param_init: xnmt.models.templates.ParamInitializer = xnmt.default_param_init,
               bias_init: xnmt.ParamInitializer = xnmt.default_bias_init,
               init_fastext: Optional[str] = None,
               weight_noise: float = xnmt.default_weight_noise,
               fix_norm: Optional[float] = None):
    super().__init__(emb_dim=emb_dim, weight_noise=weight_noise, fix_norm=fix_norm)
    # Embedding Parameters
    pcol = xnmt.param_manager(self)
    self.vocab_size = self.choose_vocab_size(vocab_size, vocab, yaml_path, src_reader, trg_reader)
    emb_mtr_dim = (self.vocab_size, self.emb_dim)

    if init_fastext is not None:
      xnmt.logger.info("Setting Dense to False because of init_fastext")
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

  def _read_fasttext_embeddings(self, vocab: xnmt.Vocab, init_fastext: str):
    """
    Reads FastText embeddings from a file. Also prints stats about the loaded embeddings for sanity checking.

    Args:
      vocab: a `Vocab` object containing the vocabulary for the experiment
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

      xnmt.logger.info(f"{in_vocab} vocabulary matches out of {total_embs} total embeddings; "
                       f"{missing} vocabulary words without a pretrained embedding out of {self.vocab_size}")

    return embeddings

  def is_batchable(self):
    return True


class BagOfWordsEmbedder(WordEmbedder, xnmt.Serializable):

  ONE_MB = 1000 * 1024

  @xnmt.serializable_init
  def __init__(self,
               emb_dim = xnmt.default_layer_dim,
               ngram_vocab: xnmt.Vocab = None,
               word_vocab: Optional[xnmt.Vocab] = xnmt.Ref("model.src_reader.vocab", default=None),
               char_vocab: Optional[xnmt.Vocab] = xnmt.Ref("model.src_reader.char_vocab", default=None),
               ngram_size: int = 1,
               transform: Optional[xnmt.models.Transform] = None,
               include_lower_ngrams: bool = True,
               weight_noise: float = xnmt.default_weight_noise,
               fix_norm: Optional[float] = None):
    super().__init__(emb_dim=emb_dim, weight_noise=weight_noise, fix_norm=fix_norm)
    self.transform = self.add_serializable_component("transform", transform,
                                                     lambda: xnmt.modules.nn.transforms.NonLinear(
                                                       input_dim=ngram_vocab.vocab_size(),
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

  def is_batchable(self):
    return False

class CharCompositionEmbedder(WordEmbedder, xnmt.Serializable):
  @xnmt.serializable_init
  def __init__(self,
               char_vocab: Optional[xnmt.structs.vocabs.CharVocab] = xnmt.Ref("model.src_reader.char_vocab", default=None),
               vocab_size: Optional[int] = None,
               emb_dim: int = xnmt.default_layer_dim,
               weight_noise: float = xnmt.default_weight_noise,
               param_init: xnmt.models.templates.ParamInitializer = xnmt.default_param_init,
               composer: models.SequenceComposer = xnmt.bare(composers.SumComposer),
               fix_norm: Optional[float] = None):
    super().__init__(emb_dim=emb_dim, weight_noise=weight_noise, fix_norm=fix_norm)
    self.composer = composer
    # Embedding Parameters
    pcol = xnmt.param_manager(self)
    self.vocab_size = self.choose_vocab_size(vocab_size, char_vocab, '', None, None)
    emb_mtr = (self.vocab_size, self.emb_dim)
    self.embeddings = pcol.add_lookup_parameters(emb_mtr,
                                                 init=param_init.initializer(emb_mtr,  is_lookup=True))
    # Model States
    self.train = False
    self.save_processed_arg("vocab_size", self.vocab_size)


  def _embed_word(self, word: sent.SegmentedWord, is_batched: bool = False):
    char_embeds = self.embeddings.batch(xnmt.mark_as_batch(word.chars))

    char_embeds = [dy.pick_batch_elem(char_embeds, i) for i in range(len(word.chars))]
    return self.composer.compose(char_embeds)

  def is_batchable(self):
    return False

class CompositeEmbedder(models.Embedder, xnmt.Serializable):
  @xnmt.serializable_init
  def __init__(self, embedders):
    self.embedders = embedders

  def embed_sent(self, x: Any):
    embeddings = [embedder.embed_sent(x) for embedder in self.embedders]
    ret = []
    for j in range(len(embeddings[0])):
      ret.append(dy.esum([embeddings[i][j] for i in range(len(embeddings))]))
    return xnmt.ExpressionSequence(expr_list=ret, mask=embeddings[0].mask)

  def embed(self, word: Any) -> dy.Expression:
    def select_word(_word, _embedder):
      if type(_word) == sent.SegmentedWord and type(_embedder) == LookupEmbedder:
        _word = _word.word
      return _word

    return dy.esum([embedder.embed(select_word(word, embedder)) for embedder in self.embedders])

  def is_batchable(self):
    return False


class NoopEmbedder(models.Embedder, xnmt.Serializable):
  """
  This embedder performs no lookups but only passes through the inputs.

  Normally, the input is a Sentence object, which is converted to an expression.

  Args:
    emb_dim: Size of the inputs
  """
  @xnmt.serializable_init
  def __init__(self, emb_dim: Optional[int]) -> None:
    self.emb_dim = emb_dim

  def embed(self, x: Union[np.ndarray, list]) -> dy.Expression:
    return dy.inputTensor(x, batched=xnmt.is_batched(x))

  def embed_sent(self, x: sent.Sentence) -> xnmt.ExpressionSequence:
    # TODO refactor: seems a bit too many special cases that need to be distinguished
    batched = xnmt.is_batched(x)
    first_sent = x[0] if batched else x
    if hasattr(first_sent, "get_array"):
      if not batched:
        return xnmt.LazyNumpyExpressionSequence(lazy_data=x.get_array())
      else:
        return xnmt.LazyNumpyExpressionSequence(lazy_data=xnmt.mark_as_batch([s for s in x]), mask=x.mask)
    else:
      if not batched:
        embeddings = [self.embed(word) for word in x]
      else:
        embeddings = []
        for word_i in range(x.sent_len()):
          embeddings.append(self.embed(xnmt.mark_as_batch([single_sent[word_i] for single_sent in x])))
      return xnmt.ExpressionSequence(expr_list=embeddings, mask=x.mask)

  def is_batchable(self):
    return True


class PositionEmbedder(models.Embedder, xnmt.Serializable):
  @xnmt.serializable_init
  def __init__(self,
               max_pos: int,
               emb_dim: int = xnmt.default_layer_dim,
               param_init: xnmt.models.templates.ParamInitializer = xnmt.default_param_init):
    """
    max_pos: largest embedded position
    emb_dim: embedding size
    param_init: how to initialize embedding matrix
    """
    self.max_pos = max_pos
    self.emb_dim = emb_dim
    param_collection = xnmt.param_manager(self)
    dim = (self.emb_dim, max_pos)
    self.embeddings = param_collection.add_parameters(dim, init=param_init.initializer(dim, is_lookup=True))

  def embed(self, word): raise NotImplementedError("Position-embedding for individual words not implemented yet.")
  def embed_sent(self, sent_len: int) -> xnmt.ExpressionSequence:
    embeddings = dy.strided_select(dy.parameter(self.embeddings), [1,1], [0,0], [self.emb_dim, sent_len])
    return xnmt.ExpressionSequence(expr_tensor=embeddings, mask=None)

  def is_batchable(self):
    return True
