import xnmt
import xnmt.models.states as states
import xnmt.models as models

import xnmt.modules.nn.embedders as embedders
import xnmt.modules.nn.transducers.recurrent as recurrent


class SentenceEncoder(xnmt.Serializable):
  @xnmt.serializable_init
  def __init__(self,
               embedder: models.Embedder = xnmt.bare(embedders.LookupEmbedder),
               seq_transducer: models.SeqTransducer = xnmt.bare(recurrent.UniLSTMSeqTransducer)):
    self.embedder = embedder
    self.seq_transducer = seq_transducer

  def encode(self, src: xnmt.Batch) -> states.EncoderState:
    embed_sent = self.embedder.embed_sent(src)

    # TODO(philip30): Add segmentation part?
    return self.seq_transducer.transduce(embed_sent)

  def shared_params(self):
    return [{".embedder.emb_dim", ".seq_transducer.input_dim"}]


