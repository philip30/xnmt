import dynet as dy


_activation = {
  "tanh": dy.tanh,
  "relu": dy.rectify,
  "sigmoid": dy.logistic,
  "elu": dy.elu,
  "selu": dy.selu,
  "ainsh": dy.asinh,
  "identity": lambda x: x
}

def dynet_activation_from_string(act_str):
  if act_str not in _activation:
    raise ValueError(f"Unknown activation {act_str}")
  return _activation[act_str]
