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

  def compose(self, embeds: Union[dy.Expression, List[dy.Expression]]) -> dy.Expression:
    if type(embeds) != list:
      return dy.sum_batches(embeds)
    else:
      return dy.esum(embeds)


class AverageComposer(models.SequenceComposer, xnmt.Serializable):
  yaml_tag = "!AverageComposer"

  @xnmt.serializable_init
  def __init__(self):
    super().__init__()

  def compose(self, embeds: Union[dy.Expression, List[dy.Expression]]) -> dy.Expression:
    if type(embeds) != list:
      return dy.mean_batches(embeds)
    else:
      return dy.average(embeds)


class MaxComposer(models.SequenceComposer, xnmt.Serializable):
  yaml_tag = "!MaxComposer"

  @xnmt.serializable_init
  def __init__(self):
    super().__init__()

  def compose(self, embeds: Union[dy.Expression, List[dy.Expression]]) -> dy.Expression:
    if type(embeds) != list:
      embeds = [dy.pick_batch_elem(embeds, i) for i in range(embeds.dim()[1])]
    return dy.emax(embeds)


class SeqTransducerComposer(models.SequenceComposer, xnmt.Serializable):
  yaml_tag = "!SeqTransducerComposer"

  @xnmt.serializable_init
  def __init__(self, seq_transducer: models.SeqTransducer = xnmt.bare(recurrent.BiLSTMSeqTransducer)):
    super().__init__()
    self.seq_transducer = seq_transducer

  def compose(self, embeds: Union[dy.Expression, List[dy.Expression]]) -> dy.Expression:
    if type(embeds) != list:
      embeds = [dy.pick_batch_elem(embeds, i) for i in range(embeds.dim()[1])]
    encoding_result = self.seq_transducer.transduce(xnmt.ExpressionSequence(expr_list=embeds))
    return encoding_result.encoder_final_states[-1].main_expr()


class DyerHeadComposer(models.SequenceComposer, xnmt.Serializable):
  yaml_tag = "!DyerHeadComposer"

  @xnmt.serializable_init
  def __init__(self,
               fwd_combinator: models.UniDiSeqTransducer = xnmt.bare(recurrent.UniLSTMSeqTransducer),
               bwd_combinator: models.UniDiSeqTransducer = xnmt.bare(recurrent.UniLSTMSeqTransducer),
               transform: models.Transform = xnmt.bare(transforms.AuxNonLinear)):
    super().__init__()
    self.fwd_combinator = fwd_combinator
    self.bwd_combinator = bwd_combinator
    self.transform = transform

  def compose(self, embeds: Union[dy.Expression, List[dy.Expression]]) -> dy.Expression:
    if type(embeds) != list:
      embeds = [dy.pick_batch_elem(embeds, i) for i in range(embeds.dim()[1])]

    fwd_state = self.fwd_combinator.initial_state()
    bwd_state = self.fwd_combinator.initial_state()
    # The embedding of the Head should be in the first element of the list
    fwd_state = fwd_state.add_input(embeds[-1])
    bwd_state = bwd_state.add_input(embeds[-1])

    for i in range(len(embeds)-1):
      fwd_state = fwd_state.add_input(embeds[i])
      bwd_state = bwd_state.add_input(embeds[-(i+1)])

    return self.transform.transform(dy.concatenate([fwd_state.output(), bwd_state.output()]))


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

  def compose(self, embeds):
    if type(embeds) != list:
      embeds = [dy.pick_batch_elem(embeds, i) for i in range(embeds.dim()[1])]

    if len(embeds) < self.ngram_size:
      embeds.extend([dy.zeros(self.embed_dim)] * (self.ngram_size-len(embeds)))

    embeds = dy.transpose(dy.concatenate([dy.concatenate_cols(embeds)], d=2), [2, 1, 0])
    embeds = dy.conv2d_bias(embeds, self.filter, self.bias, (self.embed_dim, 1))
    embeds = dy.max_dim(dy.pick(embeds, index=0), d=0)

    return self.transform.transform(embeds)

