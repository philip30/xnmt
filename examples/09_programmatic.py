# It is also possible to configure model training using Python code rather than
# YAML config files. This is less convenient and usually not necessary, but there
# may be cases where the added flexibility is needed. This basically works by
# using XNMT as a library of components that are initialized and run in this
# config file.
#
# This demonstrates a standard model training, including set up of logging, model
# saving, etc.; networks are saved into YAML files that can again be loaded using
# the standard YAML  way (examples/07_load_finetune.yaml) or the Python way
# (10_programmatic_load.py)
#
# To launch this, use ``python -m examples.09_programmatic``, making sure that XNMT
# setup.py has been run properly.


import os
import random

import numpy as np

from xnmt.modules.nn.attenders import MlpAttender
from xnmt.structs.batchers import SrcBatcher, InOrderBatcher
from xnmt.modules.nn.bridges import CopyBridge
from xnmt.modules.nn.decoders import ArbLenDecoder
from xnmt.modules.nn.embedders import LookupEmbedder
from xnmt.eval.tasks import LossEvalTask, AccuracyEvalTask
from xnmt.experiments import Experiment
from xnmt.inferences import AutoRegressiveInference
from xnmt.modules.input_readers import PlainTextReader
from xnmt.modules.transducers import BiLSTMSeqTransducer, UniLSTMSeqTransducer
from xnmt.modules.nn.transforms import AuxNonLinear
from xnmt.modules.nn.scorers import Softmax
from xnmt.modules.optimizers import AdamTrainer
from xnmt.internal.param_collections import ParamManager
from xnmt.internal.persistence import save_to_file
import xnmt.internal.tee
from xnmt.train.regimens import SimpleTrainingRegimen
from xnmt.networks.seq2seq import DefaultTranslator
from xnmt.structs.vocabs import Vocab

seed=13
random.seed(seed)
np.random.seed(seed)

EXP_DIR = os.path.dirname(__file__)
EXP = "programmatic"

model_file = f"{EXP_DIR}/networks/{EXP}.mod"
log_file = f"{EXP_DIR}/logs/{EXP}.log"

xnmt.internal.tee.set_out_file(log_file, EXP)

ParamManager.init_param_col()
ParamManager.param_col.model_file = model_file

src_vocab = Vocab(vocab_file="examples/data/head.ja.vocab")
trg_vocab = Vocab(vocab_file="examples/data/head.en.vocab")

batcher = SrcBatcher(batch_size=64)

inference = AutoRegressiveInference(batcher=InOrderBatcher(batch_size=1))

layer_dim = 512

model = DefaultTranslator(
  src_reader=PlainTextReader(vocab=src_vocab),
  trg_reader=PlainTextReader(vocab=trg_vocab),
  src_embedder=LookupEmbedder(emb_dim=layer_dim, vocab_size=len(src_vocab)),

  encoder=BiLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim, layers=1),
  attender=MlpAttender(hidden_dim=layer_dim, state_dim=layer_dim, input_dim=layer_dim),
  decoder=ArbLenDecoder(input_dim=layer_dim,
                        embedder=LookupEmbedder(emb_dim=layer_dim, vocab_size=len(trg_vocab)),
                        rnn=UniLSTMSeqTransducer(input_dim=layer_dim, hidden_dim=layer_dim,
                                                         decoder_input_dim=layer_dim, yaml_path="decoder"),
                        transform=AuxNonLinear(input_dim=layer_dim, output_dim=layer_dim,
                                                       aux_input_dim=layer_dim),
                        scorer=Softmax(vocab_size=len(trg_vocab), input_dim=layer_dim),
                        bridge=CopyBridge(dec_dim=layer_dim, dec_layers=1)),
  inference=inference
)

train = SimpleTrainingRegimen(
  name=f"{EXP}",
  model=model,
  batcher=batcher,
  trainer=AdamTrainer(alpha=0.001),
  run_for_epochs=2,
  src_file="examples/data/head.ja",
  trg_file="examples/data/head.en",
  dev_tasks=[LossEvalTask(src_file="examples/data/head.ja",
                          ref_file="examples/data/head.en",
                          model=model,
                          batcher=batcher)],
)

evaluate = [AccuracyEvalTask(eval_metrics="bleu,wer",
                             src_file="examples/data/head.ja",
                             ref_file="examples/data/head.en",
                             hyp_file=f"examples/output/{EXP}.test_hyp",
                             inference=inference,
                             model=model)]

standard_experiment = Experiment(
  name="programmatic",
  model=model,
  train=train,
  evaluate=evaluate
)

# run experiment
standard_experiment(save_fct=lambda: save_to_file(model_file, standard_experiment))

exit()
