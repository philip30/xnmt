import dynet as dy

import xnmt
import xnmt.models as models

import xnmt.modules.nn.transducers.recurrent as recurrent
import xnmt.modules.nn.transforms as transforms
from typing import List, Union, Optional


class SumComposer(models.SequenceComposer, xnmt.Serializable):
  yaml_tag = "!SumComposer"
  @xnmt.serializable_init
  def __init__(self):
    super().__init__()

  def compose(self, embeds: xnmt.ExpressionSequence) -> dy.Expression:
    tensor = embeds.as_tensor()
    tensor = embeds.mask.cmult_to_tensor_expr(tensor, inverse=True)
    return dy.sum_dim(tensor, d=[1])


class AverageComposer(models.SequenceComposer, xnmt.Serializable):
  yaml_tag = "!AverageComposer"
  @xnmt.serializable_init
  def __init__(self):
    super().__init__()

  def compose(self, embeds: xnmt.ExpressionSequence) -> dy.Expression:
    tensor = embeds.as_tensor()
    tensor = embeds.mask.cmult_to_tensor_expr(tensor, inverse=True)
    return dy.mean_dim(tensor, d=[1], b=False)


class MaxComposer(models.SequenceComposer, xnmt.Serializable):
  yaml_tag = "!MaxComposer"
  @xnmt.serializable_init
  def __init__(self):
    super().__init__()

  def compose(self, embeds: xnmt.ExpressionSequence) -> dy.Expression:
    tensor = embeds.as_tensor()
    tensor = embeds.mask.cmult_to_tensor_expr(tensor, inverse=True)
    return dy.max_dim(tensor, d=1)


class SeqTransducerComposer(models.SequenceComposer, xnmt.Serializable):
  yaml_tag = "!SeqTransducerComposer"
  @xnmt.serializable_init
  def __init__(self, seq_transducer: models.SeqTransducer = xnmt.bare(recurrent.BiLSTMSeqTransducer)):
    super().__init__()
    self.seq_transducer = seq_transducer

  def compose(self, embeds: xnmt.ExpressionSequence) -> dy.Expression:
    encoding_result = self.seq_transducer.transduce(embeds)
    return encoding_result.encoder_final_states[-1].main_expr()


class ConvolutionComposer(models.SequenceComposer, xnmt.Serializable):
  yaml_tag = "!ConvolutionComposer"
  @xnmt.serializable_init
  def __init__(self,
               ngram_size: Optional[int] = 4,
               transform: models.Transform = xnmt.bare(transforms.Cwise),
               param_init: xnmt.ParamInitializer = xnmt.default_param_init,
               bias_init: xnmt.ParamInitializer = xnmt.default_bias_init,
               embed_dim: int = xnmt.default_layer_dim,
               hidden_dim:int = xnmt.default_layer_dim):
    model = xnmt.param_manager(self)
    dim = (1, ngram_size, embed_dim, hidden_dim)
    self.filter = model.add_parameters(dim=dim, init=param_init.initializer(dim))
    self.bias = model.add_parameters(dim=(embed_dim,), init=bias_init.initializer(embed_dim))
    self.ngram_size = ngram_size
    self.embed_dim = embed_dim
    self.transform = transform

  def compose(self, embed_expr: xnmt.ExpressionSequence):
    tensor = embed_expr.as_tensor()
    tensor = embed_expr.mask.cmult_to_tensor_expr(tensor, inverse=True)
    dim = tensor.dim()
    
    if dim[0][1] < self.ngram_size:
      deficit = self.ngram_size - dim[0][1]
      tensor = dy.concatenate([tensor, dy.zeros((self.embed_dim, deficit), batch_size=dim[1])], d=1)

    embeds = dy.transpose(dy.concatenate([tensor], d=2), [2, 1, 0])
    embeds = dy.conv2d_bias(embeds, self.filter, self.bias, (self.embed_dim, 1))
    
    embeds = dy.max_dim(dy.pick(embeds, index=0), d=0)

    return self.transform.transform(embeds)

