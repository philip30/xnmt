
from typing import Any, List, Optional, Sequence

from xnmt.internal.persistence import serializable_init, Serializable

class Vocab(Serializable):
  yaml_tag = "!Vocab"
  """
  An open vocabulary that converts between strings and integer ids.

  The open vocabulary is realized via a special unknown-word token that is used whenever a word is not inside the
  list of known tokens.
  This class is immutable, i.e. its contents are not to change after the vocab has been initialized.

  For initialization, i2w or vocab_file must be specified, but not both.

  Args:
    i2w: complete list of known words, including ``<s>`` and ``</s>``.
    vocab_file: file containing one word per line, and not containing <s>, </s>, <unk>
  """

  SS = 0
  ES = 1
  PAD = 2
  UNK = 3

  
  SS_STR = "<s>"
  ES_STR = "</s>"
  UNK_STR = "<unk>"
  PAD_STR = "<pad>"

  @serializable_init
  def __init__(self,
               i2w: Optional[Sequence[str]] = None,
               vocab_file: Optional[str] = None):
    super().__init__()
    assert i2w is None or vocab_file is None
    assert i2w or vocab_file
    if vocab_file:
      i2w = self.i2w_from_vocab_file(vocab_file)
    assert i2w is not None
    self.i2w = i2w
    self.w2i = {word: word_id for (word_id, word) in enumerate(self.i2w)}
    if Vocab.UNK_STR not in self.w2i:
      self.w2i[Vocab.UNK_STR] = len(self.i2w)
      self.i2w.append(Vocab.UNK_STR)
    self.save_processed_arg("i2w", self.i2w)
    self.save_processed_arg("vocab_file", None)

  def i2w_from_vocab_file(self, vocab_file: str) -> List[str]:
    """Load the vocabulary from a file.

    If ``sentencepiece_vocab`` is set to True, this will accept a sentencepiece vocabulary file

    Args:
      vocab_file: file containing one word per line, and not containing ``<s>``, ``</s>``, ``<unk>``
    """
    reserved = [Vocab.SS_STR, Vocab.ES_STR, Vocab.PAD_STR, Vocab.UNK_STR]
    vocab = {word: i for i, word in enumerate(reserved)}
    with open(vocab_file, encoding='utf-8') as f:
      for line in f:
        word = line.strip()
        word = word.split('\t')[0]
        self._add_word_to_vocab(vocab, word)
    vocab = [word for word, word_id in sorted(vocab.items(), key=lambda x: x[1])]
    return vocab

  def _add_word_to_vocab(self, vocab, word):
    vocab[word] = len(vocab)

  def convert(self, w: str) -> int:
    return self.w2i.get(w, self.UNK)

  def __getitem__(self, i: int) -> str:
    return self.i2w[i]

  def __len__(self) -> int:
    return len(self.i2w)

  def vocab_size(self):
    return len(self.i2w)

  def is_compatible(self, other: Any) -> bool:
    """
    Check if this vocab produces the same conversions as another one.
    """
    if not isinstance(other, Vocab):
      return False
    if len(self) != len(other):
      return False
    return self.w2i == other.w2i


class CharVocab(Vocab):
  yaml_tag = "!CharVocab"

  def _add_word_to_vocab(self, vocab, word):
    for c in word:
      if c not in vocab:
        vocab[c] = len(vocab)


class SimultActionVocab(Vocab):
  yaml_tag = "!SimultActionVocab"

  PAD = 4
  UNK = 5
  SS = 6
  ES = 7
  VOCAB_SIZE = 8
  
  @serializable_init
  def __init__(self):
    self.i2w = ["READ", "WRITE", "PREDICT_READ", "PREDICT_WRITE", "PAD", Vocab.UNK_STR, Vocab.SS_STR, Vocab.ES_STR]
    self.w2i = {word: word_id for (word_id, word) in enumerate(self.i2w)}