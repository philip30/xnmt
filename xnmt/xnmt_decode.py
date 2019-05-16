import argparse, sys

import xnmt
from xnmt.eval import tasks
from xnmt.internal import param_collections, utils

def main() -> None:
  parser = argparse.ArgumentParser()
  utils.add_dynet_argparse(parser)
  parser.add_argument("--src", help=f"Path of source file to read from.", required=True)
  parser.add_argument("--hyp", help="Path of file to write hypothesis to.", required=True)
  parser.add_argument("--mod", help="Path of model file to read.", required=True)
  args = parser.parse_args()

  loaded_experiment = xnmt.load_experiment_from_path(args.mod)
  model = loaded_experiment.model
  inference = model.inference
  param_collections.ParamManager.populate()

  decoding_task = tasks.DecodingEvalTask(args.src, args.hyp, model, inference)
  decoding_task.eval()

if __name__ == "__main__":
  sys.exit(main())
