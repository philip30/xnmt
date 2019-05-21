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
    trg_vocab = xnmt.Vocab(vocab_file="examples/data/head.en.vocab")
    embedder = nn.LookupEmbedder(emb_dim=layer_dim, vocab=trg_vocab)
    rnn = nn.UniLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim, dropout=0.5)

    self.model = networks.Seq2Seq(
      src_reader=modules.EmptyTextReader(add_bos=True, vocab=trg_vocab, add_eos=False),
      trg_reader=modules.PlainTextReader(vocab=trg_vocab, add_bos=False, add_eos=True),
      encoder=nn.SeqEncoder(
        embedder=embedder,
        seq_transducer=rnn),
      decoder=nn.ArbLenDecoder(
        input_feeding=False,
        init_with_bos=False,
        input_dim=layer_dim,
        attender=None,
        embedder=embedder,
        rnn=rnn,
        transform=nn.AuxNonLinear(input_dim=layer_dim, output_dim=layer_dim, aux_input_dim=layer_dim),
        scorer=nn.Softmax(input_dim=layer_dim, vocab=trg_vocab),
        bridge=nn.CopyBridge(dec_dim=layer_dim, dec_layers=1))
    )
    xnmt.event_trigger.set_train(False)
    self.src_data = list(self.model.src_reader.read_sents("examples/data/head.en"))
    self.trg_data = list(self.model.trg_reader.read_sents("examples/data/head.en"))
    self.batcher = xnmt.structs.batchers.SrcBatcher(batch_size=2)
    self.src, self.trg = self.batcher.pack(self.src_data, self.trg_data)

  def test_single(self):
    xnmt.event_trigger.set_train(True)
    self.model.calc_nll(src=self.src[0],
                        trg=self.trg[0]).value()

    xnmt.event_trigger.set_train(False)
    self.model.generate(self.src[0], xnmt.inferences.GreedySearch())
    self.model.generate(self.src[0], xnmt.inferences.BeamSearch())


  def test_inference_forced(self):
    xnmt.event_trigger.set_train(False)
    inference = xnmt.inferences.AutoRegressiveInference(
      src_file="examples/data/head.en",
      ref_file="examples/data/head.en",
      search_strategy=xnmt.inferences.GreedySearch(is_forced=True),
    )
    inference.perform_inference(self.model)

    self.assertTrue(True)

if __name__ == '__main__':
  unittest.main()
