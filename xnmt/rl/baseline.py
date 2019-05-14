import dynet as dy

from typing import List

import xnmt.internal.persistence as persistence
import xnmt.modules.nn.transforms as transforms




class TransformBaseline(Baseline, persistence.Serializable):
  def __init__(self, transform: transforms.Transform=None):
    super().__init__()
    self.transform = self.add_serializable_component("transform", transform, lambda: transform)

  def calculate_baseline(self, input_states: List[dy.Expression]) -> List[dy.Expression]:
    transform_seq = []
    for input_state in input_states:
      transform_seq.append(self.transform.transform(dy.nobackprop(input_state)))
    return transform_seq


