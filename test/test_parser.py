import unittest

from xnmt.modules import input_readers
from xnmt.structs import vocabs, sentences


class TestCoNLLInputReader(unittest.TestCase):

  def test_read_tree(self):
    vocab = vocabs.Vocab(vocab_file="examples/data/head.en.vocab")
    reader = input_readers.CoNLLToRNNGActionsReader(vocab, vocab, None)
    tree = list(reader.read_sents(filename="examples/data/parse/head.en.conll"))
    expected = [sentences.RNNGAction(sentences.RNNGAction.Type.GEN, vocab.convert("can")),
                sentences.RNNGAction(sentences.RNNGAction.Type.GEN, vocab.convert("you")),
                sentences.RNNGAction(sentences.RNNGAction.Type.GEN, vocab.convert("do")),
                sentences.RNNGAction(sentences.RNNGAction.Type.REDUCE_LEFT),
                sentences.RNNGAction(sentences.RNNGAction.Type.REDUCE_LEFT),
                sentences.RNNGAction(sentences.RNNGAction.Type.GEN, vocab.convert("it")),
                sentences.RNNGAction(sentences.RNNGAction.Type.REDUCE_RIGHT),
                sentences.RNNGAction(sentences.RNNGAction.Type.GEN, vocab.convert("in")),
                sentences.RNNGAction(sentences.RNNGAction.Type.GEN, vocab.convert("one")),
                sentences.RNNGAction(sentences.RNNGAction.Type.GEN, vocab.convert("day")),
                sentences.RNNGAction(sentences.RNNGAction.Type.REDUCE_LEFT),
                sentences.RNNGAction(sentences.RNNGAction.Type.REDUCE_LEFT),
                sentences.RNNGAction(sentences.RNNGAction.Type.REDUCE_RIGHT),
                sentences.RNNGAction(sentences.RNNGAction.Type.GEN, vocab.convert("?")),
                sentences.RNNGAction(sentences.RNNGAction.Type.REDUCE_RIGHT),
                sentences.RNNGAction(sentences.RNNGAction.Type.GEN, vocab.ES),
                sentences.RNNGAction(sentences.RNNGAction.Type.REDUCE_RIGHT)]
    self.assertListEqual(tree[0].actions, expected)

if __name__ == '__main__':
  unittest.main()
