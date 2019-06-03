import unittest

import dynet as dy

import xnmt
import xnmt.modules as modules
import xnmt.modules.nn as nn
import xnmt.networks as networks

class TestFreeDecodingLoss(unittest.TestCase):

  def setUp(self):
    layer_dim = 4
    xnmt.refresh_internal()
    src_vocab = xnmt.Vocab(vocab_file="examples/data/head.ja.vocab")
    trg_vocab = xnmt.Vocab(vocab_file="examples/data/head.en.vocab")
    self.model = networks.Seq2Seq(
      src_reader=modules.PlainTextReader(vocab=src_vocab),
      trg_reader=modules.PlainTextReader(vocab=trg_vocab),
      encoder=nn.SeqEncoder(
        embedder=nn.LookupEmbedder(emb_dim=layer_dim, vocab=src_vocab),
        seq_transducer=nn.BiLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim)),
      decoder=nn.ArbLenDecoder(
        input_dim=layer_dim,
        attender=nn.MlpAttender(input_dim=layer_dim, state_dim=layer_dim, hidden_dim=layer_dim),
        embedder=nn.LookupEmbedder(emb_dim=layer_dim, vocab=trg_vocab),
        rnn=nn.UniLSTMSeqTransducer(input_dim=layer_dim,
                                    hidden_dim=layer_dim,
                                    decoder_input_dim=layer_dim,
                                    yaml_path=xnmt.Path("model.decoder.rnn")),
        transform=nn.NonLinear(input_dim=layer_dim*2, output_dim=layer_dim),
        scorer=nn.Softmax(input_dim=layer_dim, vocab=trg_vocab),
        bridge=nn.NoBridge(dec_dim=layer_dim))
    )
    xnmt.event_trigger.set_train(False)
    self.src_data = list(self.model.src_reader.read_sents("examples/data/head.ja"))
    self.trg_data = list(self.model.trg_reader.read_sents("examples/data/head.en"))
    self.batcher = xnmt.structs.batchers.SrcBatcher(batch_size=2)
    self.src, self.trg = self.batcher.pack(self.src_data, self.trg_data)
    self.layer_dim = layer_dim

  def test_single(self):
    self.model.generate(self.src[0], xnmt.inferences.GreedySearch())
    self.model.generate(self.src[0], xnmt.inferences.BeamSearch())

    self.model.calc_nll(src=xnmt.mark_as_batch([self.src_data[0]]),
                        trg=xnmt.mark_as_batch([self.trg_data[0]])).value()
    self.assertTrue(True)

  def test_reporting(self):
    xnmt.event_trigger.set_train(False)
    xnmt.event_trigger.set_reporting(True)
    inference = xnmt.inferences.AutoRegressiveInference(src_file="examples/data/head.ja",
                                                        ref_file="examples/data/head.en",
                                                        trg_file="test/output/hyp",
                                                        search_strategy=xnmt.inferences.GreedySearch(),
                                                        reporter=xnmt.reports.ProbReporter())
    inference.perform_inference(self.model)
    xnmt.event_trigger.set_reporting(False)
    
  def test_position_embedder(self):
    self.model.decoder.embedder.position_embedder = nn.SinCosPositionEmbedder(self.layer_dim)
    xnmt.event_trigger.set_train(True)
    mle_loss = xnmt.train.MLELoss()
    loss, loss_stat = mle_loss.calc_loss(self.model, self.src[0], self.trg[0]).compute()
    losses = []
    for s, t in zip(self.src[0], self.trg[0]):
      loss_i, _ = mle_loss.calc_loss(self.model,
                                     xnmt.mark_as_batch([s.get_unpadded_sent()]),
                                     xnmt.mark_as_batch([t.get_unpadded_sent()])).compute()
      losses.append(loss_i)

    self.assertAlmostEqual(dy.sum_batches(loss).scalar_value(), dy.esum(losses).scalar_value(), places=8)


  def test_same_loss_batch_single(self):
    xnmt.event_trigger.set_train(True)
    mle_loss = xnmt.train.MLELoss()
    loss, loss_stat = mle_loss.calc_loss(self.model, self.src[0], self.trg[0]).compute()
    losses = []
    for s, t in zip(self.src[0], self.trg[0]):
      loss_i, _ = mle_loss.calc_loss(self.model,
                                     xnmt.mark_as_batch([s.get_unpadded_sent()]),
                                     xnmt.mark_as_batch([t.get_unpadded_sent()])).compute()
      losses.append(loss_i)

    self.assertAlmostEqual(dy.sum_batches(loss).scalar_value(), dy.esum(losses).scalar_value(), places=8)


if __name__ == '__main__':
  unittest.main()
