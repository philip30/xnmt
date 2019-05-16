import dynet as dy
import numpy as np
import numbers

from typing import Sequence, Optional, Union

import xnmt.inferences as inferences
import xnmt.modules.input_readers as input_readers
import xnmt.internal.events as events
import xnmt.structs.vocabs as vocabs
import xnmt.structs.batchers as batchers
import xnmt.structs.sentences as sent
import xnmt.inferences.search_strategies as search_strategies
import xnmt.modules.nn.scorers as scorers
import xnmt.structs.losses as losses

from xnmt.internal.persistence import Serializable, bare, serializable_init

from .auto_regressive import AutoRegressiveTranslator
from xnmt.networks.seq2seq import DefaultTranslator


class EnsembleTranslator(AutoRegressiveTranslator, Serializable):
  """
  A translator that decodes from an ensemble of DefaultTranslator networks.
  Args:
    models: A list of DefaultTranslator instances; for all networks, their
      src_reader.vocab and trg_reader.vocab has to match (i.e., provide
      identical conversions to) those supplied to this class.
    src_reader: A reader for the source side.
    trg_reader: A reader for the target side.
    inference: The inference strategy used for this ensemble.
  """

  yaml_tag = '!EnsembleTranslator'

  @events.register_xnmt_handler
  @serializable_init
  def __init__(self,
               models: Sequence[DefaultTranslator],
               src_reader: input_readers.InputReader,
               trg_reader: input_readers.InputReader,
               inference: inferences.AutoRegressiveInference=bare(inferences.AutoRegressiveInference)) -> None:
    super().__init__(src_reader=src_reader, trg_reader=trg_reader)
    self.models = models
    self.inference = inference

    # perform checks to verify the networks can logically be ensembled
    for i, model in enumerate(self.models):
      if hasattr(self.src_reader, "vocab") or hasattr(model.src_reader, "vocab"):
        assert self.src_reader.vocab.is_compatible(model.src_reader.vocab), \
          f"src_reader.vocab is not compatible with model {i}"
      assert self.trg_reader.vocab.is_compatible(model.trg_reader.vocab), \
        f"trg_reader.vocab is not compatible with model {i}"

    # proxy object used for generation, to avoid code duplication
    self._proxy = DefaultTranslator(
      self.src_reader,
      self.trg_reader,
      EnsembleListDelegate([model.src_embedder for model in self.models]),
      EnsembleListDelegate([model.encoder for model in self.models]),
      EnsembleListDelegate([model.attender for model in self.models]),
      EnsembleDecoder([model.decoder for model in self.models])
    )

  def shared_params(self):
    shared = [params for model in self.models for params in model.shared_params()]
    return shared

  def set_trg_vocab(self, trg_vocab: Optional[vocabs.Vocab] = None) -> None:
    self._proxy.set_trg_vocab(trg_vocab=trg_vocab)

  def calc_nll(self, src: Union[batchers.Batch, sent.Sentence], trg: Union[batchers.Batch, sent.Sentence]) -> losses.LossExpr:
    loss_values = [model.calc_nll(src, trg).loss_value() for model in self.models]
    ret_expr = []
    ret_units = []
    for loss_expr, unit in loss_values:
      ret_expr.append(loss_expr)
      ret_units.append(unit)

    return losses.LossExpr(dy.esum(ret_expr), np.sum(ret_units))

  def generate(self,
               src: batchers.Batch,
               search_strategy: search_strategies.SearchStrategy) -> Sequence[sent.Sentence]:
    return self._proxy.generate(src, search_strategy)


class EnsembleListDelegate(object):
  """
  Auxiliary object to wrap a list of objects for ensembling.
  This class can wrap a list of objects that exist in parallel and do not need
  to interact with each other. The main functions of this class are:
  - All attribute access and function calls are delegated to the wrapped objects.
  - When wrapped objects return values, the list of all returned values is also
    wrapped in an EnsembleListDelegate object.
  - When EnsembleListDelegate objects are supplied as arguments, they are
    "unwrapped" so the i-th object receives the i-th element of the
    EnsembleListDelegate argument.
  """

  def __init__(self, objects):
    assert isinstance(objects, (tuple, list))
    self._objects = objects

  def __getitem__(self, key):
    return self._objects[key]

  def __setitem__(self, key, value):
    self._objects[key] = value

  def __iter__(self):
    return self._objects.__iter__()

  def __call__(self, *args, **kwargs):
    return self.__getattr__('__call__')(*args, **kwargs)

  def __len__(self):
    return len(self._objects)

  def __getattr__(self, attr):
    def unwrap(list_idx, args, kwargs):
      args = [arg if not isinstance(arg, EnsembleListDelegate) else arg[list_idx] \
              for arg in args]
      kwargs = {key: val if not isinstance(val, EnsembleListDelegate) else val[list_idx] \
                for key, val in kwargs.items()}
      return args, kwargs

    attrs = [getattr(obj, attr) for obj in self._objects]
    if callable(attrs[0]):
      def wrapper_func(*args, **kwargs):
        ret = []
        for i, attr_ in enumerate(attrs):
          args_i, kwargs_i = unwrap(i, args, kwargs)
          ret.append(attr_(*args_i, **kwargs_i))
        if all(val is None for val in ret):
          return None
        else:
          return EnsembleListDelegate(ret)
      return wrapper_func
    else:
      return EnsembleListDelegate(attrs)

  def __setattr__(self, attr, value):
    if not attr.startswith('_'):
      if isinstance(value, EnsembleListDelegate):
        for i, obj in enumerate(self._objects):
          setattr(obj, attr, value[i])
      else:
        for obj in self._objects:
          setattr(obj, attr, value)
    else:
      self.__dict__[attr] = value

  def __repr__(self):
    return "EnsembleListDelegate([" + ', '.join(repr(elem) for elem in self._objects) + "])"


class EnsembleDecoder(EnsembleListDelegate):
  """
  Auxiliary object to wrap a list of decoders for ensembling.
  This behaves like an EnsembleListDelegate, except that it overrides
  get_scores() to combine the individual decoder's scores.
  Currently only supports averaging.
  """
  def calc_log_probs(self, mlp_dec_states):
    scores = [obj.scorer.calc_log_probs(dec_state.as_vector()) for obj, dec_state in zip(self._objects, mlp_dec_states)]
    return dy.average(scores)

  def best_k(self, mlp_dec_states, k: numbers.Integral, normalize_scores: bool = False):
    logprobs = self.calc_log_probs(mlp_dec_states)
    return scorers.find_best_k(logprobs.npvalue(), k)
