import unittest

import dynet as dy
import xnmt
import xnmt.modules.nn as nn


class TestEncoder(unittest.TestCase):

  def setUp(self):
    xnmt.internal.events.clear()
    xnmt.internal.param_collections.ParamManager.init_param_col()
    layer_dim = 512
    src_vocab = xnmt.Vocab(vocab_file="examples/data/head.ja.vocab")
    trg_vocab = xnmt.Vocab(vocab_file="examples/data/head.en.vocab")
    self.layer_dim = layer_dim
    self.src_reader = xnmt.modules.PlainTextReader(vocab=src_vocab)
    self.trg_reader = xnmt.modules.PlainTextReader(vocab=trg_vocab)
    self.src_data = list(self.src_reader.read_sents("examples/data/head.ja"))
    self.trg_data = list(self.trg_reader.read_sents("examples/data/head.en"))
    self.model = xnmt.networks.Seq2Seq(
      encoder = nn.SeqEncoder(
        embedder = nn.LookupEmbedder(emb_dim=layer_dim, vocab_size=100),
        seq_transducer= nn.BiLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim, layers=3),
      ),
      decoder = nn.ArbLenDecoder(
        attender=nn.MlpAttender(input_dim=layer_dim, state_dim=layer_dim, hidden_dim=layer_dim),
        input_dim=layer_dim,
        embedder=nn.LookupEmbedder(emb_dim=layer_dim, vocab_size=100),
        rnn=nn.UniLSTMSeqTransducer(input_dim=layer_dim,
                                 hidden_dim=layer_dim,
                                 decoder_input_dim=layer_dim,
                                 yaml_path=xnmt.Path("model.decoder.rnn")),
        transform=nn.NonLinear(input_dim=layer_dim*2, output_dim=layer_dim),
        scorer=nn.Softmax(input_dim=layer_dim, vocab_size=100),
        bridge=nn.CopyBridge(dec_dim=layer_dim, dec_layers=1)
      ),
      src_reader=self.src_reader,
      trg_reader=self.trg_reader
    )

  def assert_in_out_len_equal(self, model: xnmt.networks.Seq2Seq):
    dy.renew_cg()
    xnmt.event_trigger.set_train(True)
    src = self.src_data[0], self.src_data[1]
    xnmt.event_trigger.start_sent(xnmt.mark_as_batch(src))
    embeddings = model.encoder.embedder.embed_sent(xnmt.mark_as_batch(src))
    encodings = model.encoder.seq_transducer.transduce(embeddings)
    self.assertEqual(len(embeddings), len(encodings.encode_seq))

  def test_bi_lstm_encoder_len(self):
    self.assert_in_out_len_equal(self.model)

  def test_uni_lstm_encoder_len(self):
    self.model.encoder.seq_transducer = nn.UniLSTMSeqTransducer(input_dim=self.layer_dim, hidden_dim=self.layer_dim)
    self.assert_in_out_len_equal(self.model)
  
  def test_res_lstm_encoder_len(self):
    self.model.encoder.seq_transducer = nn.ResidualSeqTransducer(
      input_dim=self.layer_dim,
      child=self.model.encoder.seq_transducer,
      layer_norm=True,
      dropout=0.5
    )
    self.assert_in_out_len_equal(self.model)

  def test_multihead_attention_encoder_len(self):
    self.model.encoder.seq_transducer = nn.MultiHeadAttentionSeqTransducer(input_dim=self.layer_dim)
    self.assert_in_out_len_equal(self.model)

  def test_identity(self):
    self.model.encoder.seq_transducer = nn.IdentitySeqTransducer()
    self.assert_in_out_len_equal(self.model)
    
if __name__ == '__main__':
  unittest.main()
