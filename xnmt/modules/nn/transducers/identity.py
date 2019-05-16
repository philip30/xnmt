import xnmt


class IdentitySeqTransducer(xnmt.models.SeqTransducer, xnmt.Serializable):
  yaml_tag = "!IdentitySeqTransducer"
  """
  A transducer that simply returns the input.
  """
  @xnmt.serializable_init
  def __init__(self) -> None: pass

  def transduce(self, seq) -> xnmt.models.EncoderState:
    return xnmt.models.EncoderState(seq, None)

