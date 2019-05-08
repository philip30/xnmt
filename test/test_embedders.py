import unittest
import dynet as dy

import numpy as np
import random
from itertools import islice

import xnmt.events

from xnmt.input_readers import PlainTextReader, CharFromWordTextReader
from xnmt.modelparts.embedders import LookupEmbedder, BagOfWordsEmbedder, CharCompositionEmbedder
from xnmt.seq_composer import SumComposer, SeqTransducerComposer, DyerHeadComposer
from xnmt.seq_composer import MaxComposer, AverageComposer, ConvolutionComposer
from xnmt.modelparts.transforms import NonLinear, AuxNonLinear
from xnmt.sent import SegmentedWord
from xnmt.transducers.recurrent import BiLSTMSeqTransducer, UniLSTMSeqTransducer
from xnmt.param_collections import ParamManager

from xnmt import events, event_trigger, batchers
from xnmt.vocabs import Vocab, CharVocab

class PretrainedSimpleWordEmbedderSanityTest(unittest.TestCase):
  def setUp(self):
    events.clear()
    self.input_reader = PlainTextReader(vocab=Vocab(vocab_file="examples/data/head.ja.vocab"))
    list(self.input_reader.read_sents('examples/data/head.ja'))
    ParamManager.init_param_col()

  def test_load(self):
    """
    Checks that the embeddings can be loaded, have the right dimension, and that one line matches.
    """
    embedder = LookupEmbedder(init_fastext='examples/data/wiki.ja.vec.small', emb_dim=300, vocab=self.input_reader.vocab)
    # self.assertEqual(embedder.embeddings.shape()[::-1], (self.input_reader.vocab_size(), 300))

    with open('examples/data/wiki.ja.vec.small', encoding='utf-8') as vecfile:
      test_line = next(islice(vecfile, 9, None)).split()  # Select the vector for '日'
    test_word = test_line[0]
    test_id = self.input_reader.vocab.w2i[test_word]
    test_emb = test_line[1:]

    self.assertTrue(np.allclose(embedder.embeddings.batch([test_id]).npvalue().tolist(),
                                np.array(test_emb, dtype=float).tolist(), rtol=1e-5))


class TestEmbedder(unittest.TestCase):
  def setUp(self):
    # Seeding
    np.random.seed(2)
    random.seed(2)
    layer_dim = 4
    xnmt.events.clear()
    ParamManager.init_param_col()
    self.src_vocab = Vocab(vocab_file="examples/data/head.ja.vocab")
    self.src_char_vocab = CharVocab(vocab_file="examples/data/head.ja.vocab")
    self.ngram_vocab = Vocab(vocab_file="examples/data/head.ngramcount.ja")
    self.trg_vocab = Vocab(vocab_file="examples/data/head.en.vocab")
    
    self.src_reader = CharFromWordTextReader(vocab= self.src_vocab, char_vocab= self.src_char_vocab)
    self.trg_reader = PlainTextReader(vocab=self.trg_vocab)
    
    
    self.layer_dim = layer_dim
    self.src_data = list(self.src_reader.read_sents("examples/data/head.ja"))
    self.trg_data = list(self.trg_reader.read_sents("examples/data/head.en"))
    self.src, self.trg = batchers.TrgBatcher(batch_size=3).pack(self.src_data, self.trg_data)
    dy.renew_cg(immediate_compute=True, check_validity=True)

  def test_lookup_composer(self):
    embedder = LookupEmbedder(emb_dim=self.layer_dim, vocab_size=100)
    embedder.embed_sent(self.src[1])
    embedder.embed(self.src[1][1][1])
    
  def test_sum_composer(self):
    embedder = CharCompositionEmbedder(emb_dim=self.layer_dim,
                                       composer=SumComposer(),
                                       char_vocab=self.src_char_vocab)
    embedder.embed_sent(self.src[1])

  def test_avg_composer(self):
    embedder = CharCompositionEmbedder(emb_dim=self.layer_dim,
                                       composer=AverageComposer(),
                                       char_vocab=self.src_char_vocab)
    embedder.embed_sent(self.src[1])

  def test_max_composer(self):
    embedder = CharCompositionEmbedder(emb_dim=self.layer_dim,
                                       composer=MaxComposer(),
                                       char_vocab=self.src_char_vocab)
    embedder.embed_sent(self.src[1])
    
  def test_conv_composer(self):
    composer = ConvolutionComposer(ngram_size=2,
                                   transform=NonLinear(self.layer_dim, self.layer_dim, activation="relu"),
                                   embed_dim=self.layer_dim,
                                   hidden_dim=self.layer_dim)
    embedder = CharCompositionEmbedder(emb_dim=self.layer_dim,
                                       composer=composer,
                                       char_vocab=self.src_char_vocab)
    embedder.embed_sent(self.src[1])
    
  def test_transducer_composer(self):
    composer = SeqTransducerComposer(seq_transducer=BiLSTMSeqTransducer(input_dim=self.layer_dim,
                                                                        hidden_dim=self.layer_dim))
    embedder = CharCompositionEmbedder(emb_dim=self.layer_dim,
                                       composer=composer,
                                       char_vocab=self.src_char_vocab)
    event_trigger.set_train(True)
    event_trigger.start_sent(self.src[1])
    embedder.embed_sent(self.src[1])

  def test_dyer_composer(self):
    composer = DyerHeadComposer(fwd_combinator=UniLSTMSeqTransducer(input_dim=self.layer_dim, hidden_dim=self.layer_dim),
                                bwd_combinator=UniLSTMSeqTransducer(input_dim=self.layer_dim, hidden_dim=self.layer_dim),
                                transform=AuxNonLinear(input_dim=self.layer_dim,
                                                       output_dim=self.layer_dim,
                                                       aux_input_dim=self.layer_dim))
    embedder = CharCompositionEmbedder(emb_dim=self.layer_dim,
                                       composer=composer,
                                       char_vocab=self.src_char_vocab)
    event_trigger.set_train(True)
    event_trigger.start_sent(self.src[1])
    embedder.embed_sent(self.src[1])

  def test_bagofwords_embedder(self):
    embedder = BagOfWordsEmbedder(self.layer_dim, char_vocab=self.src_char_vocab, ngram_vocab= self.ngram_vocab, ngram_size=3)
    event_trigger.set_train(True)
    event_trigger.start_sent(self.src[1])
    embedder.embed_sent(self.src[1])

  def test_bagofwords_embedder_with_word_vocab(self):
    embedder = BagOfWordsEmbedder(self.layer_dim, word_vocab=self.src_vocab, ngram_vocab= self.ngram_vocab, ngram_size=3)
    event_trigger.set_train(True)
    event_trigger.start_sent(self.src[1])
    embedder.embed_sent(self.src[1])

  def test_segmented_word(self):
    a = SegmentedWord([1,2,3], 10)
    b = SegmentedWord([1,2,3], 10)
    c = SegmentedWord([2,3,4], 10)
    d = SegmentedWord([1,2,3], 9)

    self.assertEqual(a, b)
    self.assertEqual(a, [1,2,3])
    self.assertEqual(a, 10)
    self.assertNotEqual(a, c)
    self.assertNotEqual(a, d)
    
    self.assertNotEqual(type(self.src[0][0][0]), SegmentedWord)
    self.assertEqual(type(self.src[0][0].words[0]), SegmentedWord)

#
#  def test_lookup_composer_learn(self):
#    enc = self.segmenting_encoder
#    char_vocab = Vocab(i2w=['a', 'b', 'c', 'd'])
#    enc.segment_composer = LookupComposer(
#        word_vocab = None,
#        char_vocab = char_vocab,
#        hidden_dim = self.layer_dim,
#        vocab_size = 4
#    )
#    event_trigger.set_train(True)
#    enc.segment_composer.set_word((0, 1, 2), 0, 3) # abc 0
#    enc.segment_composer.transduce([])
#    enc.segment_composer.set_word((0, 2, 1), 0, 3) # acb 1
#    enc.segment_composer.transduce([])
#    enc.segment_composer.set_word((0, 3, 2), 0, 3) # adc 2
#    enc.segment_composer.transduce([])
#    enc.segment_composer.set_word((0, 1, 2), 0, 3) # abc 0
#    enc.segment_composer.transduce([])
#    enc.segment_composer.set_word((1, 3, 2), 0, 3) # bdc 3
#    enc.segment_composer.transduce([])
#    enc.segment_composer.set_word((3, 3, 3), 0, 3) # ddd 1 -> acb is the oldest
#    enc.segment_composer.transduce([])
#    act = dict(enc.segment_composer.lrucache.items())
#    exp = {'abc': 0, 'ddd': 1, 'adc': 2, 'bdc': 3}
#    self.assertDictEqual(act, exp)
#
#    enc.segment_composer.set_word((0, 2, 1), 0, 3)
#    enc.segment_composer.transduce([])
#    enc.segment_composer.set_word((0, 3, 2), 0, 3)
#    enc.segment_composer.transduce([])
#    enc.segment_composer.set_word((0, 1, 2), 0, 3)  # abc 0
#    enc.segment_composer.transduce([])
#    enc.segment_composer.set_word((1, 3, 2), 0, 3)  # bdc 3
#    enc.segment_composer.transduce([])
#    enc.segment_composer.set_word((3, 3, 3), 0, 3)
#    enc.segment_composer.transduce([])
#    enc.segment_composer.set_word((0, 3, 1), 0, 3)
#    enc.segment_composer.transduce([])
#
#    event_trigger.set_train(False)
#    enc.segment_composer.set_word((3, 3, 2), 0, 3)
#    enc.segment_composer.transduce([])
#
#  def test_chargram_composer_learn(self):
#    enc = self.segmenting_encoder
#    char_vocab = Vocab(i2w=['a', 'b', 'c', 'd'])
#    enc.segment_composer = CharNGramComposer(
#        word_vocab = None,
#        char_vocab = char_vocab,
#        hidden_dim = self.layer_dim,
#        ngram_size = 2,
#        vocab_size = 5,
#    )
#    event_trigger.set_train(True)
#    enc.segment_composer.set_word((0, 1, 2), 0, 3) # a:0, ab:1, b: 2, bc: 3, c: 4
#    enc.segment_composer.transduce([])
#    act = dict(enc.segment_composer.lrucache.items())
#    exp = {'a': 0, 'ab': 1, 'b': 2, 'bc': 3, 'c': 4}
#    self.assertDictEqual(act, exp)
#
#    enc.segment_composer.set_word((2, 3), 0, 2) # c, cd, d
#    enc.segment_composer.transduce([])
#    act = dict(enc.segment_composer.lrucache.items())
#    exp = {'cd': 0, 'd': 1, 'b': 2, 'bc': 3, 'c': 4}
#    self.assertDictEqual(act, exp)


