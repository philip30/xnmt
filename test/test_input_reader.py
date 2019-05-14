import unittest

from xnmt.modules import input_readers
from xnmt.structs import vocabs, sentences


class TestInputReader(unittest.TestCase):

  def test_one_file_multiple_readers(self):
    vocab = vocabs.Vocab(vocab_file="examples/data/head.en.vocab")
    cr = input_readers.CompoundReader(readers=[input_readers.PlainTextReader(vocab),
                                               input_readers.LengthTextReader()])
    en_sents = list(cr.read_sents(filename="examples/data/head.en"))
    self.assertEqual(len(en_sents), 10)
    self.assertIsInstance(en_sents[0], sentences.CompoundSentence)
    self.assertEqual(" ".join([vocab.i2w[w] for w in en_sents[0].sents[0].words]), "can you do it in one day ? </s>")
    self.assertEqual(en_sents[0].sents[1].value, len("can you do it in one day ?".split()))

  def test_multiple_files_multiple_readers(self):
    vocab_en = vocabs.Vocab(vocab_file="examples/data/head.en.vocab")
    vocab_ja = vocabs.Vocab(vocab_file="examples/data/head.ja.vocab")
    cr = input_readers.CompoundReader(readers=[input_readers.PlainTextReader(vocab_en),
                                               input_readers.PlainTextReader(vocab_ja)])
    mixed_sents = list(cr.read_sents(filename=["examples/data/head.en", "examples/data/head.ja"]))
    self.assertEqual(len(mixed_sents), 10)
    self.assertIsInstance(mixed_sents[0], sentences.CompoundSentence)
    self.assertEqual(" ".join([vocab_en.i2w[w] for w in mixed_sents[0].sents[0].words]), "can you do it in one day ? </s>")
    self.assertEqual(" ".join([vocab_ja.i2w[w] for w in mixed_sents[0].sents[1].words]), "君 は １ 日 で それ が でき ま す か 。 </s>")


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
