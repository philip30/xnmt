import unittest

import dynet as dy

import xnmt
import xnmt.modules as modules
import xnmt.modules.nn as nn
import xnmt.networks as networks

class TestFreeDecodingLoss(unittest.TestCase):

  def setUp(self):
    layer_dim = 512
    xnmt.refresh_internal()
    src_vocab = xnmt.Vocab(vocab_file="examples/data/head.ja.vocab")
    trg_vocab = xnmt.Vocab(vocab_file="examples/data/head.en.vocab")
    self.model = networks.Seq2Seq(
      src_reader=modules.PlainTextReader(vocab=src_vocab),
      trg_reader=modules.PlainTextReader(vocab=trg_vocab),
      encoder=nn.SeqEncoder(
        embedder=nn.LookupEmbedder(emb_dim=layer_dim, vocab_size=100),
        seq_transducer=nn.BiLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim)),
      decoder=nn.ArbLenDecoder(
        input_dim=layer_dim,
        attender=nn.MlpAttender(input_dim=layer_dim, state_dim=layer_dim, hidden_dim=layer_dim),
        embedder=nn.LookupEmbedder(emb_dim=layer_dim, vocab_size=100),
        rnn=nn.UniLSTMSeqTransducer(input_dim=layer_dim,
                                    hidden_dim=layer_dim,
                                    decoder_input_dim=layer_dim,
                                    yaml_path=xnmt.Path("model.decoder.rnn")),
        transform=nn.NonLinear(input_dim=layer_dim*2, output_dim=layer_dim),
        scorer=nn.Softmax(input_dim=layer_dim, vocab_size=100),
        bridge=nn.CopyBridge(dec_dim=layer_dim, dec_layers=1))
    )
    xnmt.event_trigger.set_train(False)
    self.src_data = list(self.model.src_reader.read_sents("examples/data/head.ja"))
    self.trg_data = list(self.model.trg_reader.read_sents("examples/data/head.en"))

  def test_single(self):
    dy.renew_cg()
    self.model.generate(xnmt.mark_as_batch([self.src_data[0]]), xnmt.inferences.GreedySearch())
    dy.renew_cg()
    self.model.calc_nll(src=xnmt.mark_as_batch([self.src_data[0]]),
                        trg=xnmt.mark_as_batch([self.trg_data[0]])).value()
    self.assertTrue(True)

if __name__ == '__main__':
  unittest.main()
