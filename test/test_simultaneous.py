import unittest

# import dynet_config
# dynet_config.set(random_seed=3)

import numpy
import random
import dynet as dy

import xnmt.train.loss_calculators as loss_calculators
import xnmt.modules.nn.transforms as transforms

from xnmt.modules.nn.attenders import MlpAttender
from xnmt.modules.nn.bridges import NoBridge
from xnmt.modules.decoders import AutoRegressiveDecoder
from xnmt.modules.nn.embedders import LookupEmbedder
import xnmt.internal.events
from xnmt import event_trigger
from xnmt.structs import batchers
from xnmt.internal.param_collections import ParamManager
from xnmt.modules.input_readers import PlainTextReader, CompoundReader, SimultActionTextReader
from xnmt.modules.transducers import UniLSTMSeqTransducer
from xnmt.inferences.search_strategies import GreedySearch, BeamSearch
from xnmt.networks.translators.simult_translators import SimultaneousTranslator
from xnmt.modules.nn.transforms import AuxNonLinear
from xnmt.modules.nn.scorers import Softmax
from xnmt.structs.vocabs import Vocab

import xnmt.rl.policies as network


class TestSimultaneousTranslation(unittest.TestCase):
  
  def setUp(self):
    # Seeding
    numpy.random.seed(2)
    random.seed(2)
    layer_dim = 32
    xnmt.internal.events.clear()
    ParamManager.init_param_col()
    
    self.src_reader = PlainTextReader(vocab=Vocab(vocab_file="examples/data/head.ja.vocab"))
    self.trg_reader = PlainTextReader(vocab=Vocab(vocab_file="examples/data/head.en.vocab"))
    self.layer_dim = layer_dim
    self.src_data = list(self.src_reader.read_sents("examples/data/head.ja"))
    self.trg_data = list(self.trg_reader.read_sents("examples/data/head.en"))
    self.input_vocab_size = len(self.src_reader.vocab.i2w)
    self.output_vocab_size = len(self.trg_reader.vocab.i2w)
    self.loss_calculator = loss_calculators.MLELoss()
    
    self.model = SimultaneousTranslator(
      src_reader=self.src_reader,
      trg_reader=self.trg_reader,
      src_embedder=LookupEmbedder(emb_dim=layer_dim, vocab_size=self.input_vocab_size),
      encoder=UniLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim),
      attender=MlpAttender(input_dim=layer_dim, state_dim=layer_dim, hidden_dim=layer_dim),
      decoder=AutoRegressiveDecoder(input_dim=layer_dim,
                                    rnn=UniLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim,
                                                             decoder_input_dim=layer_dim, yaml_path="decoder"),
                                    transform=AuxNonLinear(input_dim=layer_dim, output_dim=layer_dim,
                                                           aux_input_dim=layer_dim),
                                    scorer=Softmax(vocab_size=self.output_vocab_size, input_dim=layer_dim),
                                    embedder=LookupEmbedder(emb_dim=layer_dim, vocab_size=self.output_vocab_size),
                                    bridge=NoBridge(dec_dim=layer_dim, dec_layers=1)),
      policy_train_oracle=False,
      policy_test_oracle=False,
      read_before_write=True,
    )
    event_trigger.set_train(True)
    

    my_batcher = batchers.TrgBatcher(batch_size=3)
    self.src, self.trg = my_batcher.pack(self.src_data, self.trg_data)
    dy.renew_cg(immediate_compute=True, check_validity=True)
  
  def test_train_nll(self):
    event_trigger.set_train(True)
    mle_loss = loss_calculators.MLELoss()
    mle_loss.calc_loss(self.model, self.src[0], self.trg[0])
    event_trigger.set_train(False)
    self.model.generate(batchers.mark_as_batch([self.src_data[0]]), GreedySearch())

  def test_simult_beam(self):
    event_trigger.set_train(False)
    mle_loss = loss_calculators.MLELoss()
    mle_loss.calc_loss(self.model, self.src[0], self.trg[0])
    self.model.generate(batchers.mark_as_batch([self.src_data[0]]), BeamSearch(beam_size=2))
   

class TestSimultTranslationWithGivenAction(unittest.TestCase):
  def setUp(self):
    # Seeding
    numpy.random.seed(2)
    random.seed(2)
    layer_dim = 32
    xnmt.internal.events.clear()
    ParamManager.init_param_col()
   
    src_vocab = Vocab(vocab_file="examples/data/head.ja.vocab")
    self.src_reader = CompoundReader(readers=[
      PlainTextReader(vocab=src_vocab),
      SimultActionTextReader()
    ], vocab=src_vocab)
    
    
    self.trg_reader = PlainTextReader(vocab=Vocab(vocab_file="examples/data/head.en.vocab"))
    self.layer_dim = layer_dim
    self.src_data = list(self.src_reader.read_sents(["examples/data/head.ja", "examples/data/simult/head.jaen.actions"]))
    self.trg_data = list(self.trg_reader.read_sents("examples/data/head.en"))
    self.input_vocab_size = len(self.src_reader.vocab.i2w)
    self.output_vocab_size = len(self.trg_reader.vocab.i2w)
    self.loss_calculator = loss_calculators.MLELoss()
    
    self.model = SimultaneousTranslator(
      src_reader=self.src_reader,
      trg_reader=self.trg_reader,
      src_embedder=LookupEmbedder(emb_dim=layer_dim, vocab_size=self.input_vocab_size),
      encoder=UniLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim),
      attender=MlpAttender(input_dim=layer_dim, state_dim=layer_dim, hidden_dim=layer_dim),
      decoder=AutoRegressiveDecoder(input_dim=layer_dim,
                                    rnn=UniLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim,
                                                             decoder_input_dim=layer_dim, yaml_path="decoder"),
                                    transform=AuxNonLinear(input_dim=layer_dim, output_dim=layer_dim,
                                                           aux_input_dim=layer_dim),
                                    scorer=Softmax(vocab_size=self.output_vocab_size, input_dim=layer_dim),
                                    embedder=LookupEmbedder(emb_dim=layer_dim, vocab_size=self.output_vocab_size),
                                    bridge=NoBridge(dec_dim=layer_dim, dec_layers=1)),
      policy_network = network.PolicyNetwork(transforms.MLP(2 * self.layer_dim, self.layer_dim, 2)),
      policy_train_oracle=True,
      policy_test_oracle=True
    )
    event_trigger.set_train(True)
    

    my_batcher = batchers.TrgBatcher(batch_size=3)
    self.src, self.trg = my_batcher.pack(self.src_data, self.trg_data)
    dy.renew_cg(immediate_compute=True, check_validity=True)
    
  def test_train_nll(self):
    event_trigger.set_train(True)
    mle_loss = loss_calculators.MLELoss()
    mle_loss.calc_loss(self.model, self.src[0], self.trg[0])
    
    pol_loss = loss_calculators.PolicyMLELoss()
    pol_loss._perform_calc_loss(self.model, self.src[0], self.trg[0])

    event_trigger.set_train(False)
    self.model.generate(batchers.mark_as_batch([self.src_data[0]]), GreedySearch())
   
  def test_train_mle_only(self):
    self.model.policy_network = None
    event_trigger.set_train(True)
    mle_loss = loss_calculators.MLELoss()
    mle_loss.calc_loss(self.model, self.src[0], self.trg[0])
    
    event_trigger.set_train(False)
    self.model.generate(batchers.mark_as_batch([self.src_data[0]]), GreedySearch())

  def test_composite(self):
    event_trigger.set_train(True)
    composite_loss = loss_calculators.CompositeLoss([loss_calculators.MLELoss(), loss_calculators.PolicyMLELoss()])
    composite_loss.calc_loss(self.model, self.src[0], self.trg[0])
    
    event_trigger.set_train(False)
    self.model.generate(batchers.mark_as_batch([self.src_data[0]]), GreedySearch())
    
if __name__ == "__main__":
  unittest.main()
