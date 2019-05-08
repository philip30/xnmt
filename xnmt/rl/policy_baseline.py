import dynet as dy

import xnmt.expression_seqs as expr_seq
import xnmt.persistence as persistence
import xnmt.modelparts.transforms as transforms


class Baseline(object):
  def calculate_baseline(self, input_state: expr_seq.ExpressionSequence) -> expr_seq.ExpressionSequence:
    pass


class TransformBaseline(Baseline, persistence.Serializable):
  def __init__(self, transform:transforms.Transform=None):
    super().__init__()
    self.transform = self.add_serializable_component("transform", transform, lambda: transform)

  def calculate_baseline(self, input_states: expr_seq.ExpressionSequence) -> expr_seq.ExpressionSequence:
    transform_seq = []
    for input_state in input_states:
      transform_seq.append(self.transform.transform(dy.nobackprop(input_state)))
    return expr_seq.ExpressionSequence(expr_list=transform_seq, mask=input_states.mask)


