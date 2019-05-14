import dynet as dy
import xnmt

class TransformSeqTransducer(xnmt.models.SeqTransducer, xnmt.Serializable):
  """
  A sequence transducer that applies a given transformation to the sequence's tensor representation

  Args:
      transform: the Transform to apply to the sequence
      downsample_by: if > 1, downsample the sequence via appropriate reshapes.
                     The transform must accept a respectively larger hidden dimension.
  """
  yaml_tag = '!TransformSeqTransducer'

  @xnmt.serializable_init
  def __init__(self, transform: xnmt.models.Transform, downsample_by: int = 1) -> None:
    self.transform = transform
    if downsample_by < 1: raise ValueError(f"downsample_by must be >=1, was {downsample_by}")
    self.downsample_by = downsample_by

  def transduce(self, src: xnmt.ExpressionSequence) -> xnmt.models.states.EncoderState:
    src_tensor = src.as_tensor()
    out_mask = src.mask
    if self.downsample_by > 1:
      assert len(src_tensor.dim()[0])==2, \
        f"Downsampling only supported for tensors of order to. Found dims {src_tensor.dim()}"
      (hidden_dim, seq_len), batch_size = src_tensor.dim()
      if seq_len % self.downsample_by != 0:
        raise ValueError(
          "For downsampling, sequence lengths must be multiples of the total reduce factor. "
          "Configure batcher accordingly.")
      src_tensor = dy.reshape(src_tensor,
                              (hidden_dim*self.downsample_by, seq_len//self.downsample_by),
                              batch_size=batch_size)
      if out_mask:
        out_mask = out_mask.lin_subsampled(reduce_factor=self.downsample_by)
    output = self.transform.transform(src_tensor)
    if self.downsample_by==1:
      if len(output.dim())!=src_tensor.dim(): # can happen with seq length 1
        output = dy.reshape(output, src_tensor.dim()[0], batch_size=src_tensor.dim()[1])
    output_seq = xnmt.ExpressionSequence(expr_tensor=output, mask=out_mask)
    final_states = [xnmt.models.FinalTransducerState(output_seq[-1])]
    return xnmt.models.states.EncoderState(output_seq, final_states)

