from typing import List

import xnmt
import xnmt.models as models
import xnmt.models.states as states


class ModularSeqTransducer(models.SeqTransducer, xnmt.Serializable):
  yaml_tag = "!ModularSeqTransducer"
  """
  A sequence transducer that stacks several :class:`xnmt.transducer.SeqTransducer` objects, all of which must
  accept exactly one argument (an :class:`expression_seqs.ExpressionSequence`) in their transduce method.

  Args:
    modules: list of SeqTransducer modules
  """
  @xnmt.serializable_init
  def __init__(self, modules: List[models.SeqTransducer]):
    self.modules = modules

  def shared_params(self):
    return [{".input_dim", ".modules.0.input_dim"}]

  def transduce(self, seq: xnmt.ExpressionSequence) -> states.EncoderState:
    input_expr = seq
    transduce_results = None
    for module in self.modules:
      transduce_results = module.transduce(input_expr)
      input_expr = transduce_results.encode_seq

    return states.EncoderState(input_expr, transduce_results.encoder_final_states)

