import unittest

from xnmt import input_readers, sent
from xnmt import vocabs

class TestInputReader(unittest.TestCase):

  def test_one_file_multiple_readers(self):
    vocab = vocabs.Vocab(vocab_file="examples/data/head.en.vocab")
    cr = input_readers.CompoundReader(readers=[input_readers.PlainTextReader(vocab),
                                               input_readers.LengthTextReader()])
    en_sents = list(cr.read_sents(filename="examples/data/head.en"))
    self.assertEqual(len(en_sents), 10)
    self.assertIsInstance(en_sents[0], sent.CompoundSentence)
    self.assertEqual(" ".join([vocab.i2w[w] for w in en_sents[0].sents[0].words]), "can you do it in one day ? </s>")
    self.assertEqual(en_sents[0].sents[1].value, len("can you do it in one day ?".split()))

  def test_multiple_files_multiple_readers(self):
    vocab_en = vocabs.Vocab(vocab_file="examples/data/head.en.vocab")
    vocab_ja = vocabs.Vocab(vocab_file="examples/data/head.ja.vocab")
    cr = input_readers.CompoundReader(readers=[input_readers.PlainTextReader(vocab_en),
                                               input_readers.PlainTextReader(vocab_ja)])
    mixed_sents = list(cr.read_sents(filename=["examples/data/head.en", "examples/data/head.ja"]))
    self.assertEqual(len(mixed_sents), 10)
    self.assertIsInstance(mixed_sents[0], sent.CompoundSentence)
    self.assertEqual(" ".join([vocab_en.i2w[w] for w in mixed_sents[0].sents[0].words]), "can you do it in one day ? </s>")
    self.assertEqual(" ".join([vocab_ja.i2w[w] for w in mixed_sents[0].sents[1].words]), "君 は １ 日 で それ が でき ま す か 。 </s>")


class TestCoNLLInputReader(unittest.TestCase):
  
  def test_read_tree(self):
    vocab = vocabs.Vocab(vocab_file="examples/data/head.en.vocab")
    reader = input_readers.CoNLLToRNNGActionsReader(vocab, vocab, None)
    tree = list(reader.read_sents(filename="examples/data/parse/head.en.conll"))
    expected = [sent.RNNGAction(sent.RNNGAction.Type.GEN, vocab.convert("can")),
                sent.RNNGAction(sent.RNNGAction.Type.GEN, vocab.convert("you")),
                sent.RNNGAction(sent.RNNGAction.Type.GEN, vocab.convert("do")),
                sent.RNNGAction(sent.RNNGAction.Type.REDUCE_LEFT),
                sent.RNNGAction(sent.RNNGAction.Type.REDUCE_LEFT),
                sent.RNNGAction(sent.RNNGAction.Type.GEN, vocab.convert("it")),
                sent.RNNGAction(sent.RNNGAction.Type.REDUCE_RIGHT),
                sent.RNNGAction(sent.RNNGAction.Type.GEN, vocab.convert("in")),
                sent.RNNGAction(sent.RNNGAction.Type.GEN, vocab.convert("one")),
                sent.RNNGAction(sent.RNNGAction.Type.GEN, vocab.convert("day")),
                sent.RNNGAction(sent.RNNGAction.Type.REDUCE_LEFT),
                sent.RNNGAction(sent.RNNGAction.Type.REDUCE_LEFT),
                sent.RNNGAction(sent.RNNGAction.Type.REDUCE_RIGHT),
                sent.RNNGAction(sent.RNNGAction.Type.GEN, vocab.convert("?")),
                sent.RNNGAction(sent.RNNGAction.Type.REDUCE_RIGHT),
                sent.RNNGAction(sent.RNNGAction.Type.GEN, vocab.ES),
                sent.RNNGAction(sent.RNNGAction.Type.REDUCE_RIGHT)]
    self.assertListEqual(tree[0].actions, expected)

if __name__ == '__main__':
  unittest.main()
