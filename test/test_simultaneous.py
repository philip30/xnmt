import unittest

import dynet as dy
import numpy
import random
import xnmt
import xnmt.modules.nn as nn


class TestSimultaneousTranslation(unittest.TestCase):

  def setUp(self):
#    dy.init(115)
#    numpy.random.seed(115)
#    random.seed(115)
    # Seeding
    layer_dim = 32
    xnmt.internal.events.clear()
    xnmt.internal.param_collections.ParamManager.init_param_col()

    self.src_reader = xnmt.modules.input_readers.SimultTextReader(
      text_reader=xnmt.modules.input_readers.PlainTextReader(vocab=xnmt.Vocab(vocab_file="examples/data/head.ja.vocab")),
      action_reader=xnmt.modules.input_readers.PlainTextReader(vocab=xnmt.structs.vocabs.SimultActionVocab())
    )
    self.trg_reader = xnmt.modules.input_readers.PlainTextReader(vocab=xnmt.Vocab(vocab_file="examples/data/head.en.vocab"))
    self.layer_dim = layer_dim
    self.src_data = list(self.src_reader.read_sents(["examples/data/head.ja", "examples/data/simult/head.jaen.actions"]))
    self.trg_data = list(self.trg_reader.read_sents("examples/data/head.en"))
    self.input_vocab_size = len(self.src_reader.vocab.i2w)
    self.output_vocab_size = len(self.trg_reader.vocab.i2w)
    self.loss_calculator = xnmt.train.MLELoss()

    self.model = xnmt.networks.SimultSeq2Seq(
      src_reader=self.src_reader,
      trg_reader=self.trg_reader,
      encoder=nn.SeqEncoder(
        embedder=nn.LookupEmbedder(emb_dim=layer_dim, vocab=self.src_reader.vocab),
        seq_transducer=nn.BiLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim)),
      decoder=nn.ArbLenDecoder(
        input_dim=layer_dim,
        attender=nn.MlpAttender(input_dim=layer_dim, state_dim=layer_dim, hidden_dim=layer_dim),
        embedder=nn.LookupEmbedder(emb_dim=layer_dim, vocab=self.trg_reader.vocab),
        rnn=nn.UniLSTMSeqTransducer(input_dim=layer_dim,
                                    hidden_dim=layer_dim,
                                    decoder_input_dim=layer_dim,
                                    yaml_path=xnmt.Path("model.decoder.rnn")),
        transform=nn.NonLinear(input_dim=layer_dim*2, output_dim=layer_dim),
        scorer=nn.Softmax(input_dim=layer_dim, vocab=self.trg_reader.vocab),
        bridge=nn.NoBridge(dec_dim=layer_dim)),
      policy_agent=xnmt.rl.agents.SimultPolicyAgent(
        oracle_in_train=True,
        oracle_in_test=True,
        trivial_read_before_write=False,
        trivial_exchange_read_write=False,
        default_layer_dim=layer_dim
      )
    )

    my_batcher = xnmt.structs.batchers.TrgBatcher(batch_size=3)
    self.src, self.trg = my_batcher.pack(self.src_data, self.trg_data)
    dy.renew_cg(immediate_compute=True, check_validity=True)

  def test_train_nll(self):
    xnmt.event_trigger.set_train(True)
    mle_loss = xnmt.train.MLELoss()
    mle_loss.calc_loss(self.model, self.src[0], self.trg[0])
    xnmt.event_trigger.set_train(False)
    result = self.model.generate(xnmt.mark_as_batch([self.src_data[0]]), xnmt.inferences.GreedySearch())
    self.assertNotEqual(len(result), 0)
    result = self.model.generate(xnmt.mark_as_batch([self.src_data[0]]), xnmt.inferences.BeamSearch())
    self.assertNotEqual(len(result), 0)
    inference = xnmt.inferences.AutoRegressiveInference(src_file=["examples/data/head.ja",
                                                                  "examples/data/simult/head.jaen.actions"],
                                                        ref_file="examples/data/head.en",
                                                        trg_file="test/output/hyp",
                                                        search_strategy=xnmt.inferences.GreedySearch(),
                                                        reporter=None)
    inference.perform_inference(self.model)

  

  def test_recurrent_agent(self):
    self.model.policy_agent.policy_network = xnmt.rl.policy_networks.RecurrentPolicyNetwork(
      scorer = nn.Softmax(self.layer_dim, 8, trg_reader=self.trg_reader),
      rnn = nn.UniLSTMSeqTransducer(input_dim=self.layer_dim, hidden_dim= self.layer_dim)
    )
    xnmt.event_trigger.set_train(True)
    mle_loss = xnmt.train.MLELoss()
    mle_loss.calc_loss(self.model, self.src[0], self.trg[0])
    xnmt.event_trigger.set_train(False)
    result = self.model.generate(xnmt.mark_as_batch([self.src_data[0]]), xnmt.inferences.GreedySearch())
    self.assertNotEqual(len(result), 0)
    result = self.model.generate(xnmt.mark_as_batch([self.src_data[0]]), xnmt.inferences.BeamSearch())
    self.assertNotEqual(len(result), 0)

  def test_read_before_write(self):
    self.model.policy_agent.trivial_read_before_write = True
    mle_loss = xnmt.train.MLELoss()
    mle_loss.calc_loss(self.model, self.src[0], self.trg[0])
    xnmt.event_trigger.set_train(False)
    result = self.model.generate(xnmt.mark_as_batch([self.src_data[0]]), xnmt.inferences.GreedySearch())
    self.assertNotEqual(len(result), 0)
    result = self.model.generate(xnmt.mark_as_batch([self.src_data[0]]), xnmt.inferences.BeamSearch())
    self.assertNotEqual(len(result), 0)

  def test_read_write_interchange(self):
    self.model.policy_agent.trivial_exchange_read_write = True
    mle_loss = xnmt.train.MLELoss()
    mle_loss.calc_loss(self.model, self.src[0], self.trg[0])
    xnmt.event_trigger.set_train(False)
    result = self.model.generate(xnmt.mark_as_batch([self.src_data[0]]), xnmt.inferences.GreedySearch())
    self.assertNotEqual(len(result), 0)
    result = self.model.generate(xnmt.mark_as_batch([self.src_data[0]]), xnmt.inferences.BeamSearch())
    self.assertNotEqual(len(result), 0)

if __name__ == "__main__":
  unittest.main()
