import dynet as dy

import xnmt
import xnmt.models.states as states


class ConvConnectedSeqTransducer(xnmt.models.SeqTransducer, xnmt.Serializable):
  yaml_tag = '!ConvConnectedSeqTransducer'
  """
    Input goes through through a first convolution in time and space, no stride,
    dimension is not reduced, then CNN layer for each frame several times
    Embedding sequence has same length as Input sequence
    """

  @xnmt.serializable_init
  def __init__(self,
               input_dim: int,
               window_receptor: int,
               output_dim: int,
               num_layers: int,
               internal_dim: int,
               non_linearity: str = 'linear') -> None:
    """
    Args:
      num_layers: num layers after first receptor conv
      input_dim: size of the inputs
      window_receptor: window for the receptor
      ouput_dim: size of the outputs
      internal_dim: size of hidden dimension, internal dimension
      non_linearity: Non linearity to apply between layers
      """

    model = xnmt.param_manager(self)
    self.input_dim = input_dim
    self.window_receptor = window_receptor
    self.internal_dim = internal_dim
    self.non_linearity = xnmt.modules.activations.dynet_activation_from_string(non_linearity)
    self.output_dim = output_dim


    normalInit = dy.NormalInitializer(0, 0.1)

    self.pConv1 = model.add_parameters(dim=(self.input_dim, self.window_receptor, 1, self.internal_dim), init=normalInit)
    self.pBias1 = model.add_parameters(dim=(self.internal_dim,))
    self.builder_layers = []

    for _ in range(num_layers):
        conv = model.add_parameters(dim=(self.internal_dim, 1, 1, self.internal_dim), init=normalInit)
        bias = model.add_parameters(dim=(self.internal_dim,))
        self.builder_layers.append((conv,bias))

    self.last_conv = model.add_parameters(dim=(self.internal_dim, 1, 1, self.output_dim), init=normalInit)
    self.last_bias = model.add_parameters(dim=(self.output_dim,))


  def transduce(self, embed_sent: xnmt.ExpressionSequence) -> states.EncoderState:
    src = embed_sent.as_tensor()

    sent_len = src.dim()[0][1]
    batch_size = src.dim()[1]
    pad_size = (self.window_receptor-1)/2 #TODO adapt it also for even window size

    pad = dy.zeros((self.input_dim,pad_size), batch_size=batch_size)
    src = dy.concatenate([pad, src, pad], d=1)
    padded_sent_len = sent_len + 2*pad_size

    src_chn = dy.reshape(src,(self.input_dim,padded_sent_len,1), batch_size=batch_size)
    cnn_layer1 = dy.conv2d_bias(src_chn, self.pConv1, self.pBias1, stride=[1,1])
    hidden_layer = dy.reshape(cnn_layer1, (self.internal_dim, sent_len, 1), batch_size=batch_size)
    hidden_layer = self.non_linearity(hidden_layer)

    for conv_hid, bias_hid in self.builder_layers:
      hidden_layer = dy.conv2d_bias(hidden_layer, conv_hid, bias_hid, stride=[1,1])
      hidden_layer = dy.reshape(hidden_layer,(self.internal_dim, sent_len, 1), batch_size=batch_size)
      hidden_layer = self.non_linearity(hidden_layer)
    output = dy.conv2d_bias(hidden_layer, self.last_conv, self.last_bias, stride=[1,1])
    output = dy.reshape(output, (sent_len, self.output_dim), batch_size=batch_size)
    output_seq = xnmt.ExpressionSequence(expr_tensor=output)
    return states.EncoderState(output_seq, [states.FinalTransducerState(output_seq[-1])])




