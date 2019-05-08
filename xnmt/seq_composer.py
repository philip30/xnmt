import dynet as dy

import xnmt.expression_seqs as expr_seq
import xnmt.persistence as persistence
import xnmt.transducers.base as transducers_base
import xnmt.transducers.recurrent as recurrent
import xnmt.modelparts.transforms as transforms
import xnmt.param_initializers as param_inits
import xnmt.param_collections as param_collections

from typing import List, Union, Optional
from xnmt.persistence import bare, Ref

class SequenceComposer(object):

  def compose(self, embeds: Union[dy.Expression, List[dy.Expression]]) -> dy.Expression:
    raise NotImplementedError()


class SumComposer(SequenceComposer, persistence.Serializable):
  yaml_tag = "!SumComposer"

  @persistence.serializable_init
  def __init__(self):
    super().__init__()

  def compose(self, embeds: Union[dy.Expression, List[dy.Expression]]) -> dy.Expression:
    if type(embeds) != list:
      return dy.sum_batches(embeds)
    else:
      return dy.esum(embeds)


class AverageComposer(SequenceComposer, persistence.Serializable):
  yaml_tag = "!AverageComposer"

  @persistence.serializable_init
  def __init__(self):
    super().__init__()

  def compose(self, embeds: Union[dy.Expression, List[dy.Expression]]) -> dy.Expression:
    if type(embeds) != list:
      return dy.mean_batches(embeds)
    else:
      return dy.average(embeds)


class MaxComposer(SequenceComposer, persistence.Serializable):
  yaml_tag = "!MaxComposer"

  @persistence.serializable_init
  def __init__(self):
    super().__init__()

  def compose(self, embeds: Union[dy.Expression, List[dy.Expression]]) -> dy.Expression:
    if type(embeds) != list:
      embeds = [dy.pick_batch_elem(embeds, i) for i in range(embeds.dim()[1])]
    return dy.emax(embeds)


class SeqTransducerComposer(SequenceComposer, persistence.Serializable):
  yaml_tag = "!SeqTransducerComposer"

  @persistence.serializable_init
  def __init__(self, seq_transducer: transducers_base.SeqTransducer = bare(recurrent.BiLSTMSeqTransducer)):
    super().__init__()
    self.seq_transducer = seq_transducer

  def compose(self, embeds: Union[dy.Expression, List[dy.Expression]]) -> dy.Expression:
    if type(embeds) != list:
      embeds = [dy.pick_batch_elem(embeds, i) for i in range(embeds.dim()[1])]
    self.seq_transducer.transduce(expr_seq.ExpressionSequence(expr_list=embeds))
    return self.seq_transducer.get_final_states()[-1].main_expr()


class DyerHeadComposer(SequenceComposer, persistence.Serializable):
  yaml_tag = "!DyerHeadComposer"

  @persistence.serializable_init
  def __init__(self,
               fwd_combinator: recurrent.UniLSTMSeqTransducer = bare(recurrent.UniLSTMSeqTransducer),
               bwd_combinator: recurrent.UniLSTMSeqTransducer = bare(recurrent.UniLSTMSeqTransducer),
               transform: transforms.Transform = bare(transforms.AuxNonLinear)):
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


class ConvolutionComposer(SequenceComposer, persistence.Serializable):
  yaml_tag = "!ConvolutionComposer"

  @persistence.serializable_init
  def __init__(self,
               ngram_size: Optional[int] = 4,
               transform: transforms.Transform = bare(transforms.Cwise),
               param_init=Ref("exp_global.param_init", default=bare(param_inits.GlorotInitializer)),
               bias_init=Ref("exp_global.bias_init", default=bare(param_inits.ZeroInitializer)),
               embed_dim=Ref("exp_global.default_layer_dim"),
               hidden_dim=Ref("exp_global.default_layer_dim")):
    model = param_collections.ParamManager.my_params(self)
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
