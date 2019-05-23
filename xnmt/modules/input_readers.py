import functools
import itertools
import ast
import numpy as np
import h5py

try:
  import sentencepiece as spm
except ImportError:
  pass

from typing import Iterator, Optional, Sequence, Union, List, Tuple

import xnmt
import xnmt.models as models
import xnmt.structs.sentences as sentences
import xnmt.structs.graph as graph
import xnmt.modules.output_processors as output_processors


class BaseTextReader(models.InputReader):
  def __init__(self, vocab: Optional[xnmt.Vocab]):
    self.vocab = vocab

  def read_sent(self, line: str, idx: int) -> xnmt.Sentence:
    """
    Convert a raw text line into an input object.

    Args:
      line: a single input string
      idx: sentence number
    Returns: a SentenceInput object for the input sentence
    """
    raise RuntimeError("Input readers must implement the read_sent function")

  def count_sents(self, filename: str) -> int:
    return len(xnmt.file_manager.request_text_file(filename))

  def read_sents(self, filename: str, filter_ids: Sequence[int] = None) -> Iterator[xnmt.Sentence]:
    """
    Args:
      filename: data file (text file)
      filter_ids:
    Returns: iterator over lines as strings (useful for subclasses to implement read_sents)
    """
    max_id = None
    if filter_ids is not None:
      max_id = max(filter_ids)
      filter_ids = set(filter_ids)

    lines = xnmt.file_manager.request_text_file(filename)
    for sent_count, line in enumerate(lines):
      if filter_ids is None or sent_count in filter_ids:
        yield self.read_sent(line=line, idx=sent_count)
      if max_id is not None and sent_count > max_id:
        break


class PlainTextReader(BaseTextReader, xnmt.Serializable):
  yaml_tag = "!PlainTextReader"
  """
  Handles the typical case of reading plain text files, with one sent per line.

  Args:
    vocab: Vocabulary to convert string tokens to integer ids. If not given, plain text will be assumed to contain
           space-separated integer ids.
    output_proc: output processors to revert the created sentences back to a readable string
  """
  @xnmt.serializable_init
  def __init__(self,
               vocab: Optional[xnmt.Vocab] = None,
               output_proc: Optional[List[models.OutputProcessor]] = None,
               add_bos: bool = False,
               add_eos: bool = True,
               shift_n: int = 0):
    super().__init__(vocab)
    self.output_procs = output_processors.get_output_processor(output_proc)
    self.add_eos = add_eos
    self.add_bos = add_bos
    self.shift_n = shift_n

  def read_sent(self, line: str, idx: int) -> sentences.SimpleSentence:
    words = []
    if self.add_bos:
      words.append(self.vocab.SS)
    words.extend([self.vocab.convert(word) for word in self.shift_word(line.strip().split())])
    if self.add_eos:
      words.append(self.vocab.ES)
    return sentences.SimpleSentence(idx=idx, words=words, vocab=self.vocab, output_procs=self.output_procs)

  def shift_word(self, words):
    if self.shift_n > 0:
      words = words[min(len(words), self.shift_n):]
    return words

  def vocab_size(self) -> int:
    return len(self.vocab)


class EmptyTextReader(PlainTextReader, xnmt.Serializable):
  yaml_tag = "!EmptyTextReader"

  @xnmt.serializable_init
  def __init__(self, vocab: Optional[xnmt.Vocab] = None, add_bos: bool = True, add_eos: bool = False):
    super().__init__(vocab=vocab, add_bos=add_bos, add_eos=add_eos)

  def read_sent(self, line: str, idx: int) -> sentences.SimpleSentence:
    return super().read_sent("", idx)


class LengthTextReader(BaseTextReader, xnmt.Serializable):
  yaml_tag = "!LengthTextReader"
  @xnmt.serializable_init
  def __init__(self, output_proc: Optional[List[xnmt.models.templates.OutputProcessor]] = None):
    super().__init__(None)
    self.output_procs = output_processors.get_output_processor(output_proc)

  def read_sent(self, line:str, idx:int) -> sentences.ScalarSentence:
    return sentences.ScalarSentence(idx=idx, value=len(line.strip().split()))


class CompoundReader(models.InputReader, xnmt.Serializable):
  yaml_tag = "!CompoundReader"
  """
  A compound reader reads inputs using several input readers at the same time.

  The resulting inputs will be of type :class:`sent.CompoundSentence`, which holds the results from the different
  readers as a tuple. Inputs can be read from different locations (if input file name is a sequence of filenames) or all
  from the same location (if it is a string). The latter can be used to read the same inputs using several input
  different readers which might capture different aspects of the input data.

  Args:
    readers: list of input readers to use
    vocab: not used by this reader, but some parent components may require access to the vocab.
  """
  @xnmt.serializable_init
  def __init__(self, readers: Sequence[models.InputReader], vocab: Optional[xnmt.Vocab] = None):
    if len(readers) < 2: raise ValueError("need at least two readers")
    self.readers = readers

    if vocab is None:
      for reader in readers:
        if hasattr(reader, "vocab"):
          vocab = getattr(reader, "vocab")
          break

    self.vocab = vocab

  def read_sents(self, filename: Union[str, Sequence[str]], filter_ids: Sequence[int] = None) \
          -> Iterator[xnmt.Sentence]:
    if isinstance(filename, str):
      filename = [filename] * len(self.readers)
    generators = [reader.read_sents(filename=cur_filename, filter_ids=filter_ids) for (reader, cur_filename) in
                     zip(self.readers, filename)]
    while True:
      try:
        sub_sents = [next(gen) for gen in generators]
        yield self.read_sent(sub_sents, sub_sents[0].idx)
      except StopIteration:
        return

  def count_sents(self, filename: str) -> int:
    return self.readers[0].count_sents(filename if isinstance(filename,str) else filename[0])

  def needs_reload(self) -> bool:
    return any(reader.needs_reload() for reader in self.readers)

  def read_sent(self, line: Sequence[xnmt.Sentence], idx: int):
    raise NotImplementedError()


class SimultTextReader(CompoundReader, xnmt.Serializable):
  yaml_tag = "!SimultTextReader"

  @xnmt.serializable_init
  def __init__(self, text_reader: PlainTextReader, action_reader: PlainTextReader, readers=None, vocab=None):
    super().__init__([text_reader, action_reader], text_reader.vocab)
    action_reader.add_bos = False
    action_reader.add_eos = False

  def read_sent(self, line: Tuple[sentences.SimpleSentence, sentences.SimpleSentence], idx: int):
    line[1].pad_token = xnmt.structs.vocabs.SimultActionVocab.PAD

    return sentences.OracleSentence(words = line[0].words,
                                    oracle = line[1],
                                    vocab = line[0].vocab,
                                    score = line[0].score,
                                    idx = line[0].idx,
                                    output_procs= line[0].output_procs,
                                    pad_token = line[0].pad_token)


class SentencePieceTextReader(BaseTextReader, xnmt.Serializable):
  yaml_tag = "!SentencePieceTextReader"
  """
  Read in text and segment it with sentencepiece. Optionally perform sampling
  for subword regularization, only at training time.
  https://arxiv.org/pdf/1804.10959.pdf
  """
  @xnmt.serializable_init
  def __init__(self,
               model_file: str,
               sample_train: bool=False,
               l: int=-1,
               alpha: float=0.1,
               vocab: Optional[xnmt.Vocab]=None):
    """
    Args:
      model_file: The sentence piece model file
      sample_train: On the training set, sample outputs
      l: The "l" parameter for subword regularization, how many sentences to sample
      alpha: The "alpha" parameter for subword regularization, how much to smooth the distribution
      vocab: The vocabulary
      output_proc: output processors to revert the created sentences back to a readable string
    """
    super().__init__(vocab)

    self.subword_model = spm.SentencePieceProcessor()
    self.subword_model.Load(model_file)
    self.sample_train = sample_train
    self.l = l
    self.alpha = alpha
    self.vocab = vocab
    self.train = False
    self.output_procs = output_processors.get_output_processor([output_processors.JoinPieceTextOutputProcessor()])

  def read_sent(self, line: str, idx: int) -> sentences.SimpleSentence:
    if self.sample_train and self.train:
      words = self.subword_model.SampleEncodeAsPieces(line.strip(), self.l, self.alpha)
    else:
      words = self.subword_model.EncodeAsPieces(line.strip())
    #words = [w.decode('utf-8') for w in words]
    return sentences.SimpleSentence(idx=idx,
                                    words=[self.vocab.convert(word) for word in words] + [self.vocab.convert(
                                 xnmt.Vocab.ES_STR)],
                                    vocab=self.vocab,
                                    output_procs=self.output_procs)

  def vocab_size(self) -> int:
    return len(self.vocab)


class RamlTextReader(BaseTextReader, xnmt.Serializable):
  yaml_tag = "!RamlTextReader"
  """
  Handles the RAML sampling, can be used on the target side, or on both the source and target side.
  Randomly replaces words according to Hamming Distance.
  https://arxiv.org/pdf/1808.07512.pdf
  https://arxiv.org/pdf/1609.00150.pdf
  """
  @xnmt.serializable_init
  def __init__(self,
               tau: Optional[float] = 1.,
               vocab: Optional[xnmt.Vocab] = None,
               output_proc: Optional[Sequence[models.OutputProcessor]] = None):
    """
    Args:
      tau: The temperature that controls peakiness of the sampling distribution
      vocab: The vocabulary
    """
    super().__init__(vocab)
    self.tau = tau
    self.output_procs = output_processors.get_output_processor(output_proc)

  def read_sent(self, line: str, idx: int) -> sentences.SimpleSentence:
    words = line.strip().split()
    if not xnmt.is_train():
      return sentences.SimpleSentence(idx=idx,
                                      words=[self.vocab.convert(word) for word in words] + [xnmt.Vocab.ES],
                                      vocab=self.vocab,
                                      output_procs=self.output_procs)
    word_ids = np.array([self.vocab.convert(word) for word in words])
    length = len(word_ids)
    logits = np.arange(length) * (-1) * self.tau
    logits = np.exp(logits - np.max(logits))
    probs = logits / np.sum(logits)
    num_words = np.random.choice(length, p=probs)
    corrupt_pos = np.random.binomial(1, p=num_words/length, size=(length,))
    num_words_to_sample = np.sum(corrupt_pos)
    sampled_words = np.random.choice(np.arange(2, len(self.vocab)), size=(num_words_to_sample,))
    word_ids[np.where(corrupt_pos==1)[0].tolist()] = sampled_words
    return sentences.SimpleSentence(idx=idx,
                                    words=list(word_ids.tolist()) + [xnmt.Vocab.ES],
                                    vocab=self.vocab,
                                    output_procs=self.output_procs)

  def needs_reload(self) -> bool:
    return True


class CharFromWordTextReader(PlainTextReader, xnmt.Serializable):
  yaml_tag = "!CharFromWordTextReader"
  """
  Read in word based corpus and turned that into SegmentedSentence.
  SegmentedSentece's words are characters, but it contains the information of the segmentation.
  """
  ONE_MB = 1000 * 1024

  @xnmt.serializable_init
  def __init__(self,
               vocab: xnmt.Vocab = None,
               char_vocab: xnmt.structs.vocabs.CharVocab = None,
               add_word_begin_marker = True,
               add_word_end_marker = True,
               output_proc: Optional[List[models.OutputProcessor]] = None,
               add_bos: bool = False,
               add_eos: bool = True,
               shift_n: int = 0):
    assert char_vocab is not None and vocab is not None
    super().__init__(vocab=vocab, add_bos=add_bos, add_eos=add_eos, shift_n=shift_n, output_proc=output_proc)
    self.char_vocab = char_vocab
    self.add_word_begin_marker = add_word_begin_marker
    self.add_word_end_marker = add_word_end_marker

  @functools.lru_cache(maxsize=ONE_MB)
  def convert_word(self, word):
    return [self.char_vocab.convert(c) for c in word]

  def read_sent(self, line: str, idx: int) -> sentences.SegmentedSentence:
    words = []
    segs = []
    offset = 0
    if self.add_bos:
      offset += 1
      segs.append(0)
      words.append(sentences.SegmentedWord(tuple([self.char_vocab.SS]), self.vocab.SS))
    for word in self.shift_word(line.strip().split()):
      chars = []
      # <SS>
      if self.add_word_begin_marker:
        offset += 1
        chars.append(self.char_vocab.SS)
      # Chars
      chars.extend(self.convert_word(word))
      offset += len(word)
      # <PAD>
      if self.add_word_end_marker:
        offset += 1
        chars.append(self.char_vocab.ES)
      # Outputs
      segs.append(offset-1)
      words.append(sentences.SegmentedWord(tuple(chars), self.vocab.convert(word)))
    # Adding EOS
    if self.add_eos:
      segs.append(segs[-1]+1)
      words.append(sentences.SegmentedWord(tuple([self.char_vocab.ES]), self.vocab.ES))
    # For segment actions
    segment = np.zeros(segs[-1]+1)
    segment[segs] = 1

    return sentences.SegmentedSentence(segment=segs, words=words, idx=idx, vocab=self.vocab, output_procs=self.output_procs)


class H5Reader(models.InputReader, xnmt.Serializable):
  yaml_tag = "!H5Reader"
  """
  Handles the case where sents are sequences of continuous-space vectors.

  The input is a ".h5" file, which can be created for example using xnmt.preproc.MelFiltExtractor

  The data items are assumed to be labeled with integers 0, 1, .. (converted to strings).

  Each data item will be a 2D matrix representing a sequence of vectors. They can
  be in either order, depending on the value of the "transpose" variable:
  * sents[sent_id][feat_ind,timestep] if transpose=False
  * sents[sent_id][timestep,feat_ind] if transpose=True

  Args:
    transpose: whether inputs are transposed or not.
    feat_from: use feature dimensions in a range, starting at this index (inclusive)
    feat_to: use feature dimensions in a range, ending at this index (exclusive)
    feat_skip: stride over features
    timestep_skip: stride over timesteps
    timestep_truncate: cut off timesteps if sequence is longer than specified value
  """
  @xnmt.serializable_init
  def __init__(self,
               transpose: bool = False,
               feat_from: Optional[int] = None,
               feat_to: Optional[int] = None,
               feat_skip: Optional[int] = None,
               timestep_skip: Optional[int] = None,
               timestep_truncate: Optional[int] = None):
    self.transpose = transpose
    self.feat_from = feat_from
    self.feat_to = feat_to
    self.feat_skip = feat_skip
    self.timestep_skip = timestep_skip
    self.timestep_truncate = timestep_truncate

  def read_sents(self, filename: str, filter_ids: Optional[Sequence[int]]=None) -> Iterator[
    sentences.ArraySentence]:
    with h5py.File(filename, "r") as hf:
      h5_keys = sorted(hf.keys(), key=lambda x: int(x))
      if filter_ids is not None:
        filter_ids = sorted(filter_ids)
        h5_keys = [h5_keys[i] for i in filter_ids]
        h5_keys.sort(key=lambda x: int(x))
      for sent_no, key in enumerate(h5_keys):
        inp = hf[key][:]
        if self.transpose:
          inp = inp.transpose()

        sub_inp = inp[self.feat_from: self.feat_to: self.feat_skip, :self.timestep_truncate:self.timestep_skip]
        if sub_inp.size < inp.size:
          inp = np.empty_like(sub_inp)
          np.copyto(inp, sub_inp)
        else:
          inp = sub_inp

        if sent_no % 1000 == 999:
          xnmt.logger.info(f"Read {sent_no+1} lines ({float(sent_no+1)/len(h5_keys)*100:.2f}%) of {filename} at {key}")
        yield sentences.ArraySentence(idx=filter_ids[sent_no] if filter_ids else sent_no, nparr=inp)

  def count_sents(self, filename: str) -> int:
    with h5py.File(filename, "r") as hf:
      l = len(hf.keys())
    return l


class NpzReader(models.InputReader, xnmt.Serializable):
  yaml_tag = "!NpzReader"
  """
  Handles the case where sents are sequences of continuous-space vectors.

  The input is a ".npz" file, which consists of multiply ".npy" files, each
  corresponding to a single sequence of continuous features. This can be
  created in two ways:
  * Use the builtin function numpy.savez_compressed()
  * Create a bunch of .npy files, and run "zip" on them to zip them into an archive.

  The file names should be named XXX_0, XXX_1, etc., where the final number after the underbar
  indicates the order of the sequence in the corpus. This is done automatically by
  numpy.savez_compressed(), in which case the names will be arr_0, arr_1, etc.

  Each numpy file will be a 2D matrix representing a sequence of vectors. They can
  be in either order, depending on the value of the "transpose" variable.
  * sents[sent_id][feat_ind,timestep] if transpose=False
  * sents[sent_id][timestep,feat_ind] if transpose=True

  Args:
    transpose: whether inputs are transposed or not.
    feat_from: use feature dimensions in a range, starting at this index (inclusive)
    feat_to: use feature dimensions in a range, ending at this index (exclusive)
    feat_skip: stride over features
    timestep_skip: stride over timesteps
    timestep_truncate: cut off timesteps if sequence is longer than specified value
  """
  @xnmt.serializable_init
  def __init__(self,
               transpose: bool = False,
               feat_from: Optional[int] = None,
               feat_to: Optional[int] = None,
               feat_skip: Optional[int] = None,
               timestep_skip: Optional[int] = None,
               timestep_truncate: Optional[int] = None):
    self.transpose = transpose
    self.feat_from = feat_from
    self.feat_to = feat_to
    self.feat_skip = feat_skip
    self.timestep_skip = timestep_skip
    self.timestep_truncate = timestep_truncate

  def read_sents(self, filename: str, filter_ids: Optional[Sequence[int]] = None) -> None:
    npz_file = np.load(filename, mmap_mode=None if filter_ids is None else "r")
    npz_keys = sorted(npz_file.files, key=lambda x: int(x.split('_')[-1]))
    if filter_ids is not None:
      filter_ids = sorted(filter_ids)
      npz_keys = [npz_keys[i] for i in filter_ids]
      npz_keys.sort(key=lambda x: int(x.split('_')[-1]))
    for sent_no, key in enumerate(npz_keys):
      inp = npz_file[key]
      if self.transpose:
        inp = inp.transpose()

      sub_inp = inp[self.feat_from: self.feat_to: self.feat_skip, :self.timestep_truncate:self.timestep_skip]
      if sub_inp.size < inp.size:
        inp = np.empty_like(sub_inp)
        np.copyto(inp, sub_inp)
      else:
        inp = sub_inp

      if sent_no % 1000 == 999:
        xnmt.logger.info(f"Read {sent_no+1} lines ({float(sent_no+1)/len(npz_keys)*100:.2f}%) of {filename} at {key}")
      yield sentences.ArraySentence(idx=filter_ids[sent_no] if filter_ids else sent_no, nparr=inp)
    npz_file.close()

  def count_sents(self, filename: str) -> int:
    npz_file = np.load(filename, mmap_mode="r")  # for counting sentences, only read the index
    l = len(npz_file.files)
    npz_file.close()
    return l


class GraphReader(BaseTextReader):
  def __init__(self, node_vocab, edge_vocab, value_vocab):
    super().__init__(value_vocab)
    self._node_vocab = node_vocab
    self._edge_vocab = edge_vocab
    self._value_vocab = value_vocab

  @property
  def node_vocab(self):
    return self._node_vocab

  @property
  def edge_vocab(self):
    return self._edge_vocab

  @property
  def value_vocab(self):
    return self._value_vocab


class CoNLLToRNNGActionsReader(GraphReader, xnmt.Serializable):
  yaml_tag = "!CoNLLToRNNGActionsReader"
  """
  Handles the reading of CoNLL File Format:

  ID FORM LEMMA POS FEAT HEAD DEPREL

  A single line represents a single edge of dependency parse tree.
  """
  @xnmt.serializable_init
  def __init__(self, surface_vocab: xnmt.Vocab, nt_vocab: xnmt.Vocab, edg_vocab: xnmt.Vocab, output_procs = None):
    super().__init__(nt_vocab, edg_vocab, surface_vocab)
    self.output_procs = output_processors.get_output_processor(output_procs)

  def read_sents(self, filename: str, filter_ids: Sequence[int] = None):
    # Routine to add tree
    idx = 0
    lines = []
    # Loop all lines in the file
    for line in xnmt.file_manager.request_text_file(filename):
      line = line.strip()
      if len(line) <= 1:
        yield self.emit_tree(idx, lines)
        lines.clear()
        idx += 1
      else:
        try:
          node_id, form, lemma, pos, feat, head, deprel = line.strip().split("\t")
          lines.append((int(node_id), form, lemma, pos, feat, int(head), deprel))
        except ValueError:
          xnmt.logger.error("Bad line: %s", line)
          raise
    if len(lines) != 0:
      yield self.emit_tree(idx, lines)


  def emit_tree(self, idx, lines):
    nodes = {}
    edge_list = []
    max_node = -1
    for node_id, form, lemma, pos, feat, head, deprel in lines:
      nodes[node_id] = sentences.SyntaxTreeNode(node_id=node_id, value=form, head=pos)
      max_node = max(max_node, node_id)
    nodes[max_node+1] = sentences.SyntaxTreeNode(node_id=max_node + 1, value=xnmt.Vocab.ES_STR, head=xnmt.Vocab.ES_STR)
    root = -1
    for node_id, form, lemma, pos, feat, head, deprel in lines:
      if head == 0:
        root =node_id
      else:
        edge_list.append(graph.HyperEdge(head, [node_id], None, deprel))
    edge_list.append(graph.HyperEdge(root, [max_node+1], None, xnmt.Vocab.ES_STR))
    return sentences.DepTreeRNNGSequenceSentence(idx,
                                                 score=None,
                                                 graph=graph.HyperGraph(edge_list, nodes),
                                                 surface_vocab=self.value_vocab,
                                                 nt_vocab=self.node_vocab,
                                                 edge_vocab=self.edge_vocab,
                                                 all_surfaces=True,
                                                 output_procs=self.output_procs)

class PennTreeBankReader(GraphReader, xnmt.Serializable):
  yaml_tag = "!PennTreeBankReader"
  @xnmt.serializable_init
  def __init__(self, word_vocab: xnmt.Vocab, head_vocab: xnmt.Vocab, output_procs = None):
    super().__init__(head_vocab, None, word_vocab)
    self.output_procs = output_processors.get_output_processor(output_procs)

  def _read_tree_from_line(self, line):
    stack = []
    edges = []
    nodes = {}
    now_depth = 0
    now_id = 0
    for token in line.split():
      # Process "("
      if token.startswith("("):
        stack.append([now_depth, sentences.SyntaxTreeNode(now_id, None, token[1:], sentences.SyntaxTreeNode.Type.NT)])
        nodes[now_id] = stack[-1][1]
        now_id += 1
        now_depth += 1
      else:
        try:
          end_idx = token.index(")")
        except IndexError:
          end_idx = len(token)
        if end_idx != 0:
          stack.append([now_depth, sentences.SyntaxTreeNode(now_id, token[:end_idx], None, sentences.SyntaxTreeNode.Type.T)])
          nodes[now_id] = stack[-1][1]
          now_id += 1
        # Process ")"
        for _ in range(end_idx, len(token)):
          depth, child = stack.pop()
          children = [child]
          while len(stack) > 0 and stack[-1][0] == depth:
            children.append(stack.pop()[1])
          if len(stack) > 0:
            parent = stack[-1][1]
            for child in children:
              edges.append(graph.HyperEdge(parent.node_id, [child.node_id]))
          now_depth -= 1
    return graph.HyperGraph(edges, nodes)

  def read_sents(self, filename: str, filter_ids: Sequence[int] = None):
    for idx, line in enumerate(xnmt.file_manager.request_text_file(filename)):
      yield sentences.ParseTreeRNNGSequenceSentence(idx=idx,
                                                    score=None,
                                                    graph=self._read_tree_from_line(line.strip()),
                                                    surface_vocab=self.value_vocab,
                                                    nt_vocab=self.node_vocab,
                                                    edge_vocab=self.edge_vocab,
                                                    all_surfaces=False,
                                                    output_procs=self.output_procs)


class LatticeReader(GraphReader, xnmt.Serializable):
  yaml_tag = "!LatticeReader"
  """
  Reads lattices from a text file.

  The expected lattice file format is as follows:
  * 1 line per lattice
  * lines are serialized python lists / tuples
  * 2 lists per lattice:
    - list of nodes, with every node a 4-tuple: (lexicon_entry, fwd_log_prob, marginal_log_prob, bwd_log_prob)
    - list of arcs, each arc a tuple: (node_id_start, node_id_end)
            - node_id references the nodes and is 0-indexed
            - node_id_start < node_id_end
  * All paths must share a common start and end node, i.e. <s> and </s> need to be contained in the lattice

  A simple example lattice:
    [('<s>', 0.0, 0.0, 0.0), ('buenas', 0, 0.0, 0.0), ('tardes', 0, 0.0, 0.0), ('</s>', 0.0, 0.0, 0.0)],[(0, 1), (1, 2), (2, 3)]

  Args:
    vocab: Vocabulary to convert string tokens to integer ids. If not given, plain text will be assumed to contain
           space-separated integer ids.
    text_input: If ``True``, assume a standard text file as input and convert it to a flat lattice.
    flatten: If ``True``, convert to a flat lattice, with all probabilities set to 1.
  """
  @xnmt.serializable_init
  def __init__(self, vocab: xnmt.Vocab, text_input: bool = False, flatten = False, output_procs = None):
    super().__init__(None, None, vocab)
    self.text_input = text_input
    self.flatten = flatten
    self.output_procs = output_procs

  def read_sent(self, line, idx):
    edge_list = []
    if self.text_input:
      # Node List
      nodes = [sentences.LatticeNode(node_id=0, value=xnmt.Vocab.SS)]
      for i, word in enumerate(line.strip().split()):
        nodes.append(sentences.LatticeNode(node_id=i + 1, value=self.value_vocab.convert(word)))
      nodes.append(sentences.LatticeNode(node_id=len(nodes), value=xnmt.Vocab.ES))
      # Flat edge list
      for i in range(len(nodes)-1):
        edge_list.append(graph.HyperEdge(i, [i+1]))
    else:
      node_list, arc_list = ast.literal_eval(line)
      nodes = [sentences.LatticeNode(node_id=i,
                                     value=self.value_vocab.convert(item[0]),
                                     fwd_log_prob=item[1], marginal_log_prob=item[2], bwd_log_prob=item[3])
               for i, item in enumerate(node_list)]
      if self.flatten:
        for i in range(len(nodes)-1):
          edge_list.append(graph.HyperEdge(i, [i+1]))
          nodes[i].reset_prob()
        nodes[-1].reset_prob()
      else:
        for from_index, to_index in arc_list:
          edge_list.append(graph.HyperEdge(from_index, [to_index]))

      assert nodes[0].value == self.value_vocab.SS and nodes[-1].value == self.value_vocab.ES
    # Construct graph
    ret_graph = graph.HyperGraph(edge_list, {node.node_id: node for node in nodes})
    assert len(ret_graph.roots()) == 1 # <SOS>
    assert len(ret_graph.leaves()) == 1 # <EOS>
    # Construct LatticeSentence
    return sentences.GraphSentence(idx=idx, graph=ret_graph, value_vocab=self.value_vocab, score=None)

  def vocab_size(self):
    return len(self.value_vocab)


###### A utility function to read a parallel corpus
def read_parallel_corpus(src_reader: models.InputReader,
                         trg_reader: models.InputReader,
                         src_file: str,
                         trg_file: str,
                         batcher: xnmt.structs.batchers.Batcher=None,
                         sample_sents: Optional[int] = None,
                         max_num_sents: Optional[int] = None,
                         max_src_len: Optional[int] = None,
                         max_trg_len: Optional[int] = None) -> tuple:
  """
  A utility function to read a parallel corpus.

  Args:
    src_reader:
    trg_reader:
    src_file:
    trg_file:
    batcher:
    sample_sents: if not None, denote the number of sents that should be randomly chosen from all available sents.
    max_num_sents: if not None, read only the first this many sents
    max_src_len: skip pair if src side is too long
    max_trg_len: skip pair if trg side is too long

  Returns:
    A tuple of (src_data, trg_data, src_batches, trg_batches) where ``*_batches = *_data`` if ``batcher=None``
  """
  src_data = []
  trg_data = []
  if sample_sents:
    xnmt.logger.info(f"Starting to read {sample_sents} parallel sentences of {src_file} and {trg_file}")
    src_len = src_reader.count_sents(src_file)
    trg_len = trg_reader.count_sents(trg_file)
    if src_len != trg_len: raise RuntimeError(f"training src sentences don't match trg sentences: {src_len} != {trg_len}!")
    if max_num_sents and max_num_sents < src_len: src_len = trg_len = max_num_sents
    filter_ids = np.random.choice(src_len, sample_sents, replace=False)
  else:
    xnmt.logger.info(f"Starting to read {src_file} and {trg_file}")
    filter_ids = None
    src_len, trg_len = 0, 0
  src_train_iterator = src_reader.read_sents(src_file, filter_ids)
  trg_train_iterator = trg_reader.read_sents(trg_file, filter_ids)
  for src_sent, trg_sent in itertools.zip_longest(src_train_iterator, trg_train_iterator):
    if src_sent is None or trg_sent is None:
      raise RuntimeError(f"training src sentences don't match trg sentences: {src_len or src_reader.count_sents(src_file)} != {trg_len or trg_reader.count_sents(trg_file)}!")
    if max_num_sents and (max_num_sents <= len(src_data)):
      break
    src_len_ok = max_src_len is None or src_sent.sent_len() <= max_src_len
    trg_len_ok = max_trg_len is None or trg_sent.sent_len() <= max_trg_len
    if src_len_ok and trg_len_ok:
      src_data.append(src_sent)
      trg_data.append(trg_sent)

  xnmt.logger.info(f"Done reading {src_file} and {trg_file}. Packing into batches.")

  # Pack batches
  if batcher is not None:
    src_batches, trg_batches = batcher.pack(src_data, trg_data)
  else:
    src_batches, trg_batches = src_data, trg_data

  xnmt.logger.info(f"Done packing batches.")

  return src_data, trg_data, src_batches, trg_batches
