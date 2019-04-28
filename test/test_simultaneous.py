import unittest

# import dynet_config
# dynet_config.set(random_seed=3)

import numpy
import random
import dynet as dy

import xnmt.loss_calculators as loss_calculators

from xnmt.modelparts.attenders import MlpAttender
from xnmt.modelparts.bridges import NoBridge
from xnmt.modelparts.decoders import AutoRegressiveDecoder
from xnmt.modelparts.embedders import SimpleWordEmbedder
import xnmt.events
from xnmt import batchers, event_trigger
from xnmt.param_collections import ParamManager
from xnmt.input_readers import PlainTextReader, CompoundReader, SimultActionTextReader
from xnmt.transducers.recurrent import UniLSTMSeqTransducer
from xnmt.search_strategies import GreedySearch, BeamSearch
from xnmt.simultaneous.simult_translators import SimultaneousTranslator
from xnmt.modelparts.transforms import AuxNonLinear
from xnmt.modelparts.scorers import Softmax
from xnmt.vocabs import Vocab
from xnmt.rl.policy_gradient import PolicyGradient


class TestSimultaneousTranslation(unittest.TestCase):
  
  def setUp(self):
    # Seeding
    numpy.random.seed(2)
    random.seed(2)
    layer_dim = 32
    xnmt.events.clear()
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
      src_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=self.input_vocab_size),
      encoder=UniLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim),
      attender=MlpAttender(input_dim=layer_dim, state_dim=layer_dim, hidden_dim=layer_dim),
      decoder=AutoRegressiveDecoder(input_dim=layer_dim,
                                    rnn=UniLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim,
                                                             decoder_input_dim=layer_dim, yaml_path="decoder"),
                                    transform=AuxNonLinear(input_dim=layer_dim, output_dim=layer_dim,
                                                           aux_input_dim=layer_dim),
                                    scorer=Softmax(vocab_size=self.output_vocab_size, input_dim=layer_dim),
                                    embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=self.output_vocab_size),
                                    bridge=NoBridge(dec_dim=layer_dim, dec_layers=1)),
    )
    event_trigger.set_train(True)
    

    my_batcher = batchers.TrgBatcher(batch_size=3)
    self.src, self.trg = my_batcher.pack(self.src_data, self.trg_data)
    dy.renew_cg(immediate_compute=True, check_validity=True)
  
  def test_train_nll(self):
    event_trigger.set_train(True)
    mle_loss = loss_calculators.MLELoss()
    mle_loss.calc_loss(self.model, self.src[0], self.trg[0])

  def test_simult_greedy(self):
    event_trigger.set_train(False)
    self.model.generate(batchers.mark_as_batch([self.src_data[0]]), GreedySearch())

  def test_simult_beam(self):
    event_trigger.set_train(False)
    self.model.generate(batchers.mark_as_batch([self.src_data[0]]), BeamSearch())
   
  def test_policy(self):
    event_trigger.set_train(True)
    self.model.policy_learning = PolicyGradient(input_dim=3*self.layer_dim)
    mle_loss = loss_calculators.MLELoss()
    loss = mle_loss.calc_loss(self.model, self.src[0], self.trg[0])
    event_trigger.calc_reinforce_loss(self.trg[0], self.model, loss)


class TestSimultTranslationWithGivenAction(unittest.TestCase):
  def setUp(self):
    # Seeding
    numpy.random.seed(2)
    random.seed(2)
    layer_dim = 32
    xnmt.events.clear()
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
      src_embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=self.input_vocab_size),
      encoder=UniLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim),
      attender=MlpAttender(input_dim=layer_dim, state_dim=layer_dim, hidden_dim=layer_dim),
      decoder=AutoRegressiveDecoder(input_dim=layer_dim,
                                    rnn=UniLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim,
                                                             decoder_input_dim=layer_dim, yaml_path="decoder"),
                                    transform=AuxNonLinear(input_dim=layer_dim, output_dim=layer_dim,
                                                           aux_input_dim=layer_dim),
                                    scorer=Softmax(vocab_size=self.output_vocab_size, input_dim=layer_dim),
                                    embedder=SimpleWordEmbedder(emb_dim=layer_dim, vocab_size=self.output_vocab_size),
                                    bridge=NoBridge(dec_dim=layer_dim, dec_layers=1)),
      policy_learning=PolicyGradient(input_dim=3*self.layer_dim),
      is_pretraining=True
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
    pol_loss.calc_loss(self.model, self.src[0], self.trg[0])
  
  def test_composite(self):
    composite_loss = loss_calculators.CompositeLoss([loss_calculators.MLELoss(), loss_calculators.PolicyMLELoss()])
    composite_loss.calc_loss(self.model, self.src[0], self.trg[0])
    
if __name__ == "__main__":
  unittest.main()
