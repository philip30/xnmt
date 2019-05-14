import unittest

# import dynet_config
# dynet_config.set(random_seed=3)

import numpy
import random
import dynet as dy

import xnmt.modules.input_readers as input_readers
import xnmt.internal.events
import xnmt.modules.nn.composers as composer

from xnmt.modules.nn.attenders import MlpAttender
from xnmt.modules.nn.bridges import NoBridge
from xnmt.modules.nn.decoders import RNNGDecoder
from xnmt.modules.nn.embedders import LookupEmbedder
from xnmt import event_trigger
from xnmt.structs import batchers
from xnmt.internal.param_collections import ParamManager
from xnmt.modules.transducers import UniLSTMSeqTransducer
from xnmt.modules.transducers import IdentitySeqTransducer
from xnmt.networks.translators.seq2seq import DefaultTranslator
from xnmt.inferences.search_strategies import GreedySearch, BeamSearch
from xnmt.train.loss_calculators import MLELoss
from xnmt.modules.nn.transforms import AuxNonLinear
from xnmt.structs.vocabs import Vocab


class TestGraphToGraph(unittest.TestCase):
  
  def setUp(self):
    # Seeding
    numpy.random.seed(2)
    random.seed(2)
    layer_dim = 32
    xnmt.internal.events.clear()
    ParamManager.init_param_col()
    
    edge_vocab = Vocab(vocab_file="examples/data/parse/head.en.edge_vocab")
    node_vocab = Vocab(vocab_file="examples/data/parse/head.en.node_vocab")
    value_vocab = Vocab(vocab_file="examples/data/head.en.vocab")
    
    self.src_reader = input_readers.PlainTextReader(vocab=value_vocab)
    self.trg_reader = input_readers.CoNLLToRNNGActionsReader(surface_vocab=value_vocab,
                                                             nt_vocab=node_vocab,
                                                             edg_vocab=edge_vocab)
    
    self.layer_dim = layer_dim
    self.src_data = list(self.src_reader.read_sents("examples/data/head.en"))
    self.trg_data = list(self.trg_reader.read_sents("examples/data/parse/head.en.conll"))
    self.loss_calculator = MLELoss()
    self.head_composer = composer.DyerHeadComposer(
      fwd_combinator=UniLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim),
      bwd_combinator=UniLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim),
      transform=AuxNonLinear(input_dim=layer_dim, aux_input_dim=layer_dim, output_dim=layer_dim)
    )
  
    self.model = DefaultTranslator(
      src_reader=self.src_reader,
      trg_reader=self.trg_reader,
      src_embedder=LookupEmbedder(emb_dim=layer_dim, vocab_size=len(value_vocab)),
      encoder=IdentitySeqTransducer(),
      attender=MlpAttender(input_dim=layer_dim, state_dim=layer_dim, hidden_dim=layer_dim),
      decoder=RNNGDecoder(input_dim=layer_dim,
                          rnn=UniLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim,
                                                   decoder_input_dim=layer_dim),
                          transform=AuxNonLinear(input_dim=layer_dim, output_dim=layer_dim, aux_input_dim=layer_dim),
                          bridge=NoBridge(dec_dim=layer_dim, dec_layers=1),
                          graph_reader=self.trg_reader,
                          head_composer=self.head_composer)
    )
    event_trigger.set_train(True)

    my_batcher = batchers.TrgBatcher(batch_size=1)
    self.src, self.trg = my_batcher.pack(self.src_data, self.trg_data)
    dy.renew_cg(immediate_compute=True, check_validity=True)
  
  def test_train_nll(self):
    event_trigger.set_train(True)
    event_trigger.start_sent(self.src[0])
    mle_loss = MLELoss()
    mle_loss.calc_loss(self.model, self.src[0], self.trg[0])

  def test_rnng_greedy(self):
    event_trigger.set_train(False)
    self.model.generate(batchers.mark_as_batch([self.src_data[0]]), GreedySearch())
   
  def test_rnng_beam(self):
    event_trigger.set_train(False)
    self.model.generate(batchers.mark_as_batch([self.src_data[0]]), BeamSearch())
   
if __name__ == "__main__":
  unittest.main()
