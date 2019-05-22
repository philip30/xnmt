import unittest

import dynet as dy
import random
import numpy
import xnmt
import xnmt.modules.nn as nn


class TestSimultaneousTranslation(unittest.TestCase):

  def setUp(self):
    random.seed(7)
    numpy.random.seed(7)
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
        seq_transducer=nn.UniLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim)),
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

  
  def test_same_loss_batch_single(self):
    xnmt.event_trigger.set_train(True)
    self.model.policy_agent.policy_network = None
    self.model.train_pol_mle = False
    mle_loss = xnmt.train.MLELoss()
    for src, trg in zip(self.src, self.trg):
      loss, loss_stat = mle_loss.calc_loss(self.model, src, trg).compute()
      losses = []
      for s, t in zip(src, trg):
        loss_i, _ = mle_loss.calc_loss(self.model,
                                       xnmt.mark_as_batch([s.get_unpadded_sent()]),
                                       xnmt.mark_as_batch([t.get_unpadded_sent()])).compute()
        losses.append(loss_i)
      
      self.assertAlmostEqual(dy.sum_batches(loss).scalar_value(), dy.esum(losses).scalar_value(), places=1)
    
  def test_same_loss_batch_single_pol(self):
    xnmt.event_trigger.set_train(True)
    self.model.train_nmt_mle = False
    mle_loss = xnmt.train.MLELoss()
    for src, trg in zip(self.src, self.trg):
      loss, loss_stat = mle_loss.calc_loss(self.model, src, trg).compute()
      losses = []
      for s, t in zip(src, trg):
        loss_i, _ = mle_loss.calc_loss(self.model,
                                       xnmt.mark_as_batch([s.get_unpadded_sent()]),
                                       xnmt.mark_as_batch([t.get_unpadded_sent()])).compute()
        losses.append(loss_i)
      
      self.assertAlmostEqual(dy.sum_batches(loss).scalar_value(), dy.esum(losses).scalar_value(), places=0)


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


if __name__ == "__main__":
  unittest.main()
