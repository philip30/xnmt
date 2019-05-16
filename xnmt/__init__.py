import os
import sys
import warnings

# No support for python2
if sys.version_info[0] == 2:
  raise RuntimeError("XNMT does not support python2 any longer.")

package_dir = os.path.dirname(os.path.abspath(__file__))
if package_dir not in sys.path:
  sys.path.append(package_dir)

# Setting up logging
import logging
logger = logging.getLogger('xnmt')
yaml_logger = logging.getLogger('yaml')
file_logger = logging.getLogger('xnmt_file')

# matplotlib
import matplotlib
matplotlib.use('Agg')

with warnings.catch_warnings():
  warnings.simplefilter("ignore", lineno=36)
  import h5py


# Setting up initial components
# The order does matter.
# Change with due care.
import xnmt.internal
from xnmt.internal.persistence import Serializable, serializable_init, bare, Ref, Path, load_experiment_from_path
from xnmt.internal.events import handle_xnmt_event, register_xnmt_handler
from xnmt.internal import utils
from xnmt.internal.settings import settings
import xnmt.internal.file_manager as file_manager

import xnmt.structs
from xnmt.structs.batch import Batch, Mask, mark_as_batch, is_batched
from xnmt.structs.vocabs import Vocab
from xnmt.structs.expression_seqs import ExpressionSequence, ReversedExpressionSequence, LazyNumpyExpressionSequence
from xnmt.structs.losses import LossExpr, FactoredLossExpr
from xnmt.structs.sentences import Sentence
from xnmt.structs.batchers import Batcher

ref_src_reader = Ref("model.src_reader", default=None)
ref_trg_reader = Ref("model.trg_reader", default=None)
default_layer_dim = Ref("exp_global.default_layer_dim")
default_weight_noise = Ref("exp_global.weight_noise", default=0.0)
default_dropout = Ref("exp_global.dropout", default=0.0)
param_manager = lambda x: xnmt.internal.param_collections.ParamManager.my_params(x)

import xnmt.globals
from xnmt.globals import is_train

import xnmt.models
from xnmt.models.templates import ParamInitializer
from xnmt.models import OutputProcessor

import xnmt.param_initializers
import xnmt.plotting

default_param_init = xnmt.Ref("exp_global.param_init", default=xnmt.bare(xnmt.param_initializers.GlorotInitializer))
default_bias_init = xnmt.Ref("exp_global.bias_init", default=xnmt.bare(xnmt.param_initializers.ZeroInitializer))

import xnmt.modules
import xnmt.event_trigger
import xnmt.train
import xnmt.eval
import xnmt.inferences
import xnmt.preproc
import xnmt.experiments
import xnmt.rl
import xnmt.reports
import xnmt.networks


resolved_serialize_params = {}

def init_representer(dumper, obj):
  if id(obj) not in resolved_serialize_params:
  # if len(resolved_serialize_params)==0:
    serialize_params = obj.serialize_params
  else:
    serialize_params = resolved_serialize_params[id(obj)]
  return dumper.represent_mapping('!' + obj.__class__.__name__, serialize_params)

import yaml
seen_yaml_tags = set()
for serializable_child in Serializable.__subclasses__():
  if hasattr(serializable_child, "yaml_tag") and serializable_child.yaml_tag is not None and\
      serializable_child.yaml_tag == f"{serializable_child.__name__}":
    logger.warning("Setting yaml_tag manually is obsolete:",
                   f"{serializable_child.__module__}.{serializable_child.__name__}", serializable_child.yaml_tag)
  setattr(serializable_child, "yaml_tag", f"!{serializable_child.__name__}")
  assert serializable_child.yaml_tag not in seen_yaml_tags, \
    f"encountered naming conflict: more than one class with yaml_tag='{serializable_child.yaml_tag}'. " \
    f"Change to a unique class name."
  assert getattr(serializable_child.__init__, "uses_serializable_init",
                 False), f"{serializable_child.__module__}.{serializable_child.__name__}.__init__() must be wrapped in @serializable_init."
  seen_yaml_tags.add(serializable_child.yaml_tag)
  yaml.add_representer(serializable_child, init_representer)

logger.info(f"Initialized XNMT with {len(list(Serializable.__subclasses__()))} serializable modules.")
