import functools
import collections
import numpy as np
import dynet as dy
from typing import Any, Optional, List

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
               position_embedder: Optional[models.PositionEmbedder] = None,
               fix_norm: Optional[float] = None):
    self.fix_norm = fix_norm
    self.weight_noise = weight_noise
    self.emb_dim = emb_dim
    self.position_embedder = position_embedder

  def embed(self, x: xnmt.Batch, position: Optional[int] = None) -> dy.Expression:
    """
    Embed a single word in a sentence.
    """
    ret = self._embed_word(x, xnmt.is_batched(x))
    ## Applying Fix normalization
    if self.fix_norm is not None:
      ret = dy.cdiv(ret, dy.l2_norm(ret)) * self.fix_norm
    ## Weight noise only when training
    if xnmt.globals.is_train() and self.weight_noise > 0.0:
      ret = dy.noise(ret, self.weight_noise)
    if self.position_embedder is not None and position is not None:
      ret = ret + self.position_embedder.embed_position(position)
    return ret

  def embed_sent(self, x: xnmt.Batch):
    embeddings = []
    for word_i in range(x.sent_len()):
      batch = [single_sent[word_i] for single_sent in x]
      embeddings.append(self.embed(xnmt.mark_as_batch(batch), position=word_i))
    return xnmt.ExpressionSequence(expr_list=embeddings, mask=x.mask)

  def _embed_word(self, word, is_batched):
    raise NotImplementedError()


class LookupEmbedder(WordEmbedder, transforms.Linear, xnmt.Serializable):
  yaml_tag = "!LookupEmbedder"
  @xnmt.serializable_init
  def __init__(self,
               emb_dim: int = xnmt.default_layer_dim,
               vocab_size: Optional[int] = None,
               vocab: Optional[xnmt.Vocab] = None,
               yaml_path: xnmt.Path = xnmt.Path(''),
               position_embedder: Optional[models.PositionEmbedder] = None,
               src_reader: Optional[models.InputReader] = xnmt.ref_src_reader,
               trg_reader: Optional[models.InputReader] = xnmt.ref_trg_reader,
               is_dense: bool = False,
               param_init: xnmt.models.templates.ParamInitializer = xnmt.default_param_init,
               bias_init: xnmt.ParamInitializer = xnmt.default_bias_init,
               init_fastext: Optional[str] = None,
               weight_noise: float = xnmt.default_weight_noise,
               fix_norm: Optional[float] = None):
    super().__init__(emb_dim=emb_dim, weight_noise=weight_noise,
                     position_embedder=position_embedder, fix_norm=fix_norm)
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
      if type(word[0]) == sent.SegmentedWord:
        word = [w.word for w in word]
      embedding = dy.pick_batch(self.embeddings, word) if self.is_dense else self.embeddings.batch(word)
    else:
      if isinstance(word, sent.SegmentedWord):
        word = word.word
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


class BagOfWordsEmbedder(WordEmbedder, xnmt.Serializable):
  yaml_tag = "!BagOfWordsEmbedder"

  ONE_MB = 1000 * 1024

  @xnmt.serializable_init
  def __init__(self,
               emb_dim = xnmt.default_layer_dim,
               ngram_vocab: xnmt.Vocab = None,
               word_vocab: Optional[xnmt.Vocab] = xnmt.Ref("model.src_reader.vocab", default=None),
               char_vocab: Optional[xnmt.Vocab] = xnmt.Ref("model.src_reader.char_vocab", default=None),
               position_embedder: Optional[models.PositionEmbedder] = None,
               ngram_size: int = 1,
               transform: Optional[xnmt.models.Transform] = None,
               include_lower_ngrams: bool = True,
               weight_noise: float = xnmt.default_weight_noise,
               fix_norm: Optional[float] = None):
    super().__init__(emb_dim=emb_dim, weight_noise=weight_noise, fix_norm=fix_norm, position_embedder=position_embedder)
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
        ngram = self.ngram_vocab.convert(chars[i:j+1])
        if ngram != self.ngram_vocab.UNK:
          word_vector[ngram] += 1
    
    if len(word_vector) == 0:
      word_vector[self.ngram_vocab.UNK] = 1

    return dict(word_vector)

  def _embed_word(self, word: xnmt.Batch, is_batched):
    if self.word_vocab is not None:
      ngram_stats = [self.to_ngram_stats(w.word) for w in word]
    elif self.char_vocab is not None:
      ngram_stats = [self.to_ngram_stats(w.chars) for w in word]
    else:
      raise ValueError("Either word vocab or char vocab should not be None")

    keys = [(x, i) for i in range(len(ngram_stats)) for x in ngram_stats[i].keys()]
    keys = tuple(map(list, list(zip(*keys))))
    values = [x for i in range(len(ngram_stats)) for x in ngram_stats[i].values()]
    dim = self.ngram_vocab.vocab_size(), word.batch_size()
    input_tensor = dy.sparse_inputTensor(keys, values, dim, batched=True)
    # Note: If one wants to use CHARAGRAM embeddings, use NonLinear with Relu.
    return self.transform.transform(input_tensor)


class CharCompositionEmbedder(WordEmbedder, xnmt.Serializable):
  yaml_tag = "!CharCompositionEmbedder"
  @xnmt.serializable_init
  def __init__(self,
               char_vocab: Optional[xnmt.structs.vocabs.CharVocab] = xnmt.Ref("model.src_reader.char_vocab", default=None),
               position_embedder: Optional[models.PositionEmbedder] = None,
               vocab_size: Optional[int] = None,
               emb_dim: int = xnmt.default_layer_dim,
               weight_noise: float = xnmt.default_weight_noise,
               param_init: xnmt.models.templates.ParamInitializer = xnmt.default_param_init,
               composer: models.SequenceComposer = xnmt.bare(composers.SumComposer),
               fix_norm: Optional[float] = None):
    super().__init__(emb_dim=emb_dim, weight_noise=weight_noise, fix_norm=fix_norm, position_embedder=position_embedder)
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


  def _embed_word(self, word: xnmt.Batch, is_batched: bool = False):
    assert type(word[0]) == sent.SegmentedWord
    char_inps = [w.chars for w in word]
    max_len = max([len(x) for x in char_inps])
    batch_size = len(char_inps)
    
    mask = np.zeros((batch_size, max_len), dtype=int)
    content = np.ones((batch_size, max_len), dtype=int) * xnmt.Vocab.PAD
    for i, ch in enumerate(char_inps):
      deficit = max_len - len(ch)
      if deficit > 0:
        mask[i][-deficit:] = 1
      content[i][:len(ch)] = ch
    
    char_embeds = dy.concatenate([self.embeddings.batch(ch) for ch in content.transpose()], d=1)
    return self.composer.compose(xnmt.ExpressionSequence(expr_tensor=char_embeds, mask=xnmt.Mask(mask)))


class CompositeEmbedder(models.Embedder, xnmt.Serializable):
  yaml_tag = "!CompositeEmbedder"
  @xnmt.serializable_init
  def __init__(self, embedders: List[models.Embedder]):
    self.embedders = embedders

  def embed_sent(self, x: xnmt.Batch) -> xnmt.ExpressionSequence:
    embeddings = [embedder.embed_sent(x).as_tensor() for embedder in self.embedders]
    return xnmt.ExpressionSequence(expr_tensor=dy.esum(embeddings), mask=x.mask)

  def embed(self, word: Any, position: Optional[int] = None) -> dy.Expression:
    return dy.esum([embedder.embed(word, position) for embedder in self.embedders])


class SinCosPositionEmbedder(models.PositionEmbedder, xnmt.Serializable):
  yaml_tag = "!SinCosPositionEmbedder"
  
  @xnmt.serializable_init
  def __init__(self, embed_dim: int = xnmt.default_layer_dim):
    self.embed_dim = embed_dim
    
  def embed_position(self, position: int):
    if type(position) == int:
      position = [position]
    even_flag = xnmt.Mask(np.expand_dims(np.asarray([1 if x % 2 == 0 else 0 for x in position]), axis=0).transpose())
    scale = [10000 ** ((pos - (pos%2)) / self.embed_dim) for pos in position]
    position = dy.inputTensor(position, batched=True)
    scale = dy.inputTensor(scale, batched=True)
    
    angle = dy.cdiv(position, scale)
    
    sin_pos = dy.sin(angle)
    cos_pos = dy.cos(angle)
    
    return even_flag.cmult_by_timestep_expr(sin_pos, 0) + \
           even_flag.cmult_by_timestep_expr(cos_pos, 0, inverse=True)
    