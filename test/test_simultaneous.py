import unittest

import dynet as dy
import random
import numpy
import xnmt
import xnmt.modules.nn as nn


class TestSimultaneousTranslationRRWW(unittest.TestCase):

  def setUp(self):
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
    self.src_data = list(self.src_reader.read_sents(["examples/data/head.ja", "examples/data/simult/head.jaen.rrww.actions"]))
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
                                    yaml_path=xnmt.Path("model.decoder.rnn"),
                                    decoder_input_feeding=True),
        transform=nn.NonLinear(input_dim=layer_dim*2, output_dim=layer_dim),
        scorer=nn.Softmax(input_dim=layer_dim, vocab=self.trg_reader.vocab),
        bridge=nn.ZeroBridge(dec_dim=layer_dim)),
      policy_agent=xnmt.rl.agents.SimultPolicyAgent(
        oracle_in_train=True,
        oracle_in_test=True,
        default_layer_dim=layer_dim
      ),
      default_layer_dim=layer_dim
    )

    self.seq2seq = xnmt.networks.Seq2Seq(
      src_reader=self.src_reader,
      trg_reader=self.trg_reader,
      encoder=self.model.encoder,
      decoder=self.model.decoder
    )

    my_batcher = xnmt.structs.batchers.TrgBatcher(batch_size=3, break_ties_randomly=False)
    self.src, self.trg = my_batcher.pack(self.src_data, self.trg_data)
    self.special_src, self.special_trg = my_batcher.pack([self.src_data[8], self.src_data[6], self.src_data[2]],
                                                         [self.trg_data[8], self.trg_data[6], self.trg_data[2]])

    dy.renew_cg(immediate_compute=True, check_validity=True)
    xnmt.event_trigger.set_train(False)

  def test_train_nll(self):
    xnmt.event_trigger.set_train(True)
    mle_loss = xnmt.train.MLELoss()
    mle_loss.calc_loss(self.model, self.src[0], self.trg[0])
    xnmt.event_trigger.set_train(False)
    result = self.model.generate(xnmt.mark_as_batch([self.src_data[0]]), xnmt.inferences.GreedySearch())
    self.assertNotEqual(result, self.trg_data[0].sent_len())
    result = self.model.generate(xnmt.mark_as_batch([self.src_data[0]]), xnmt.inferences.BeamSearch())
    self.assertNotEqual(result, self.trg_data[0].sent_len())
    inference = xnmt.inferences.AutoRegressiveInference(src_file=["examples/data/head.ja",
                                                                  "examples/data/simult/head.jaen.actions"],
                                                        ref_file="examples/data/head.en",
                                                        trg_file="test/output/hyp",
                                                        search_strategy=xnmt.inferences.GreedySearch(),
                                                        reporter=None)
    inference.perform_inference(self.model)
    xnmt.event_trigger.set_train(False)

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

      self.assertAlmostEqual(dy.sum_batches(loss).scalar_value(), dy.esum(losses).scalar_value(), places=4)

  def test_loss_equal_to_seq2seq(self):
    xnmt.event_trigger.set_train(False)
    self.model.policy_agent.policy_network = None
    self.model.train_pol_mle = False
    mle_loss = xnmt.train.MLELoss()
    for src, trg in zip(self.special_src, self.special_trg):
      loss, loss_stat = mle_loss.calc_loss(self.model, src, trg).compute()
      sloss, sloss_stat = mle_loss.calc_loss(self.seq2seq, src, trg).compute()
      numpy.testing.assert_array_almost_equal(loss.npvalue(), sloss.npvalue(), decimal=8)

  def test_same_loss_batch_single_pol(self):
    xnmt.event_trigger.set_train(False)
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
      self.assertAlmostEqual(dy.sum_batches(loss).scalar_value(), dy.esum(losses).scalar_value(), places=4)


class TestSimultaneousTranslationOracle(unittest.TestCase):

  def setUp(self):
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
        bridge=nn.ZeroBridge(dec_dim=layer_dim)),
      policy_agent=xnmt.rl.agents.SimultPolicyAgent(
        oracle_in_train=True,
        oracle_in_test=True,
        default_layer_dim=layer_dim
      ),
      default_layer_dim=layer_dim
    )

    my_batcher = xnmt.structs.batchers.TrgBatcher(batch_size=3, break_ties_randomly=False)
    self.src, self.trg = my_batcher.pack(self.src_data, self.trg_data)
    self.special_src, self.special_trg = xnmt.structs.batchers.InOrderBatcher(3).pack(
      [self.src_data[8], self.src_data[2], self.src_data[6]],
      [self.trg_data[8], self.trg_data[2], self.trg_data[6]]
    )


    dy.renew_cg(immediate_compute=True, check_validity=True)
    xnmt.event_trigger.set_train(False)

  def test_train_nll(self):
    xnmt.event_trigger.set_train(True)
    mle_loss = xnmt.train.MLELoss()
    mle_loss.calc_loss(self.model, self.src[0], self.trg[0])
    xnmt.event_trigger.set_train(False)
    result = self.model.generate(xnmt.mark_as_batch([self.src_data[0]]), xnmt.inferences.GreedySearch())
#    self.assertEqual(result[0].sent_len(), self.trg_data[0].sent_len())
    result = self.model.generate(xnmt.mark_as_batch([self.src_data[0]]), xnmt.inferences.BeamSearch())
#    self.assertEqual(result[0].sent_len(), self.trg_data[0].sent_len())
    inference = xnmt.inferences.AutoRegressiveInference(src_file=["examples/data/head.ja",
                                                                  "examples/data/simult/head.jaen.actions"],
                                                        ref_file="examples/data/head.en",
                                                        trg_file="test/output/hyp",
                                                        search_strategy=xnmt.inferences.GreedySearch(),
                                                        reporter=None)
    inference.perform_inference(self.model)
    xnmt.event_trigger.set_train(False)


  def test_same_loss_batch_single(self):
    #xnmt.event_trigger.set_train(True)
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

      self.assertAlmostEqual(dy.sum_batches(loss).scalar_value(), dy.esum(losses).scalar_value(), places=4)

  def test_same_loss_batch_single_pol(self):
    #xnmt.event_trigger.set_train(True)
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

      self.assertAlmostEqual(dy.sum_batches(loss).scalar_value(), dy.esum(losses).scalar_value(), places=4)

  def test_generate_pol(self):
    xnmt.event_trigger.set_train(False)
    self.model.policy_agent.oracle_in_test = False
    self.model.generate(self.src[0], xnmt.inferences.GreedySearch())

  def test_sanity(self):
    state = xnmt.rl.agents.SimultSeqLenUniDirectionalState(
      src=xnmt.mark_as_batch([xnmt.structs.sentences.SimpleSentence([3,3,3])]),
      full_encodings=xnmt.ExpressionSequence(expr_tensor=dy.zeros((32, 3), batch_size=1)),
      num_reads=[3]
    )
    pol_action = xnmt.models.SearchAction(
      action_id=[xnmt.rl.agents.SimultPolicyAgent.READ],
      log_softmax=dy.inputTensor([(1,1,1,1,1,1)], batched=True)
    )

    action = self.model.policy_agent.check_sanity(state, pol_action)
    self.assertEqual(action.action_id[0], self.model.policy_agent.WRITE)

#  def test_attention_agent(self):
#    self.model.policy_agent = xnmt.rl.agents.SimultPolicyAttentionAgent(
#      self.model.policy_agent.input_transform,
#      self.model.policy_agent.policy_network,
#      self.model.policy_agent.oracle_in_train,
#      self.model.policy_agent.oracle_in_test,
#      self.model.policy_agent.default_layer_dim,
#      nn.DotAttender(True, self.layer_dim),
#      nn.DotAttender(True, self.layer_dim),
#      nn.Linear(self.layer_dim, self.layer_dim, False),
#      nn.Linear(self.layer_dim, self.layer_dim, False),
#      nn.Linear(self.layer_dim, self.layer_dim, False),
#      nn.Linear(self.layer_dim, self.layer_dim, False),
#      nn.Linear(self.layer_dim, self.layer_dim, False),
#      nn.Linear(self.layer_dim, self.layer_dim, False)
#    )
#    self.model.train_pol_mle = True
#    self.model.policy_agent.policy_network = xnmt.rl.policy_networks.RecurrentPolicyNetwork(
#      transform = nn.NonLinear(self.layer_dim, self.layer_dim),
#      scorer = nn.Softmax(self.layer_dim, 10, trg_reader=self.trg_reader, softmax_mask=[0,1,2,3,6,7,8,9]),
#      rnn = nn.UniLSTMSeqTransducer(input_dim=self.layer_dim, hidden_dim= self.layer_dim)
#    )
#
#    mle_loss = xnmt.train.MLELoss()
#    for src, trg in zip(self.src, self.trg):
#      loss, loss_stat = mle_loss.calc_loss(self.model, src, trg).compute()
#      losses = []
#      for s, t in zip(src, trg):
#        loss_i, _ = mle_loss.calc_loss(self.model,
#                                       xnmt.mark_as_batch([s.get_unpadded_sent()]),
#                                       xnmt.mark_as_batch([t.get_unpadded_sent()])).compute()
#        losses.append(loss_i)
#
#      self.assertAlmostEqual(dy.sum_batches(loss).scalar_value(), dy.esum(losses).scalar_value(), places=4)



class TestSimultaneousTranslationPredict(unittest.TestCase):

  def setUp(self):
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
    self.src_data = list(self.src_reader.read_sents(["examples/data/head.ja", "examples/data/simult/head.jaen.lm.actions"]))
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
        bridge=nn.ZeroBridge(dec_dim=layer_dim)),
      policy_agent=xnmt.rl.agents.SimultPolicyAgent(
        oracle_in_train=True,
        oracle_in_test=True,
        default_layer_dim=layer_dim
      ),
      default_layer_dim=layer_dim
    )

    my_batcher = xnmt.structs.batchers.TrgBatcher(batch_size=3, break_ties_randomly=False)
    self.src, self.trg = my_batcher.pack(self.src_data, self.trg_data)
    dy.renew_cg(immediate_compute=True, check_validity=True)
    xnmt.event_trigger.set_train(False)

  def test_train_nll(self):
    xnmt.event_trigger.set_train(True)
    mle_loss = xnmt.train.MLELoss()
    mle_loss.calc_loss(self.model, self.src[0], self.trg[0])
    xnmt.event_trigger.set_train(False)
    result = self.model.generate(xnmt.mark_as_batch([self.src_data[0]]), xnmt.inferences.GreedySearch())
#    self.assertEqual(result[0].sent_len(), self.trg_data[0].sent_len())
    result = self.model.generate(xnmt.mark_as_batch([self.src_data[0]]), xnmt.inferences.BeamSearch())
#    self.assertEqual(result[0].sent_len(), self.trg_data[0].sent_len())
    inference = xnmt.inferences.AutoRegressiveInference(src_file=["examples/data/head.ja",
                                                                  "examples/data/simult/head.jaen.lm.actions"],
                                                        ref_file="examples/data/head.en",
                                                        trg_file="test/output/hyp",
                                                        search_strategy=xnmt.inferences.GreedySearch(),
                                                        reporter=None)
    inference.perform_inference(self.model)


  def test_train_nll_no_oracle(self):
    self.model.policy_agent.oracle_in_train = False
    self.model.policy_agent.oracle_in_test = False
    self.model.train_nmt_mle = True
    self.model.train_pol_mle = False
    xnmt.event_trigger.set_train(True)
    mle_loss = xnmt.train.MLELoss()
    mle_loss.calc_loss(self.model, self.src[0], self.trg[0])
    xnmt.event_trigger.set_train(False)
    mle_loss.calc_loss(self.model, self.src[0], self.trg[0])
    result = self.model.generate(xnmt.mark_as_batch([self.src_data[0]]), xnmt.inferences.GreedySearch())
#    self.assertEqual(result[0].sent_len(), self.trg_data[0].sent_len())
    result = self.model.generate(xnmt.mark_as_batch([self.src_data[0]]), xnmt.inferences.BeamSearch())
#    self.assertEqual(result[0].sent_len(), self.trg_data[0].sent_len())
    inference = xnmt.inferences.AutoRegressiveInference(src_file=["examples/data/head.ja",
                                                                  "examples/data/simult/head.jaen.lm.actions"],
                                                        ref_file="examples/data/head.en",
                                                        trg_file="test/output/hyp",
                                                        search_strategy=xnmt.inferences.GreedySearch(),
                                                        reporter=None)
    inference.perform_inference(self.model)

  def test_train_reinforce(self):
    xnmt.event_trigger.set_train(True)
    self.model.bleu_score_only_reward = False
    self.model.policy_agent.oracle_in_train = False
    self.model.len_reward = False
    reinf_loss = xnmt.train.ReinforceLoss(num_sample=1, max_len=20, dagger_eps=0.0)
    for src, trg in zip(self.src, self.trg):
      loss, loss_stat = reinf_loss.calc_loss(self.model, src, trg).compute()
      losses = []
      for s, t in zip(src, trg):
        s = xnmt.mark_as_batch([s.get_unpadded_sent()])
        t = xnmt.mark_as_batch([t.get_unpadded_sent()])
        loss_i, _ = reinf_loss.calc_loss(self.model, s, t).compute()
        losses.append(loss_i)

      self.assertAlmostEqual(dy.sum_batches(loss).scalar_value(), dy.esum(losses).scalar_value(), places=4)


  def test_report(self):
    xnmt.event_trigger.set_train(False)
    xnmt.event_trigger.set_reporting(True)
    self.model.policy_agent.oracle_in_test = False
    self.model.policy_agent.policy_network.scorer.softmax_mask = [0,1,2,3,6,7,8,9]
    inference = xnmt.inferences.AutoRegressiveInference(src_file=["examples/data/head.ja",
                                                                  "examples/data/simult/head.jaen.lm.actions"],
                                                        ref_file="examples/data/head.en",
                                                        trg_file="test/output/hyp",
                                                        search_strategy=xnmt.inferences.GreedySearch(),
                                                        reporter=xnmt.reports.SimultActionReporter())
    inference.perform_inference(self.model)
    xnmt.event_trigger.set_reporting(False)

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

      self.assertAlmostEqual(dy.sum_batches(loss).scalar_value(), dy.esum(losses).scalar_value(), places=4)

  def test_permute(self):
    self.model.permute = 0.5
    xnmt.event_trigger.set_train(True)
    self.model.policy_agent.policy_network = None
    self.model.train_pol_mle = False
    mle_loss = xnmt.train.MLELoss()
    for src, trg in zip(self.src, self.trg):
      loss, loss_stat = mle_loss.calc_loss(self.model, src, trg).compute()


  def test_word_scheduled_sampling(self):
    self.model.word_scheduled_sampling = 1.0
    xnmt.event_trigger.set_train(True)
    self.model.policy_agent.policy_network = None
    self.model.train_pol_mle = False
    mle_loss = xnmt.train.MLELoss()
    for src, trg in zip(self.src, self.trg):
      loss, loss_stat = mle_loss.calc_loss(self.model, src, trg).compute()


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

      self.assertAlmostEqual(dy.sum_batches(loss).scalar_value(), dy.esum(losses).scalar_value(), places=4)

  def test_generate_pol(self):
    xnmt.event_trigger.set_train(False)
    self.model.policy_agent.oracle_in_test = False
    self.model.generate(self.src[0], xnmt.inferences.GreedySearch())


  def test_recurrent_agent(self):
    self.model.policy_agent.policy_network = xnmt.rl.policy_networks.RecurrentPolicyNetwork(
      transform = nn.NonLinear(self.layer_dim, self.layer_dim),
      scorer = nn.Softmax(self.layer_dim, 10, trg_reader=self.trg_reader),
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
