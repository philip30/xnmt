import argparse
import sys

import xnmt
import xnmt.eval as evals

def main() -> None:
  parser = argparse.ArgumentParser()
  xnmt.utils.add_dynet_argparse(parser)
  parser.add_argument("--metric",
                      help=f"Scoring metric(s), a string. "
                           f"Accepted metrics are {', '.join(evals.eval_shortcuts.keys())}."
                           f"Alternatively, metrics with non-default settings can by used by specifying a Python "
                           f"Evaluator object to be parsed using eval(). Example: 'WEREvaluator(case_sensitive=True)'",
                      nargs="+")
  parser.add_argument("--hyp", help="Path to read hypothesis file from")
  parser.add_argument("--ref", help="Path to read reference file from", nargs="+")
  args = parser.parse_args()

  evaluators = args.metric
  evaluators = [evals.eval_shortcuts[shortcut]() if shortcut in evals.eval_shortcuts else eval(shortcut) for shortcut in evaluators]

  scores = evals.evaluate(args.ref, args.hyp, evaluators)
  for score in scores:
    print(score)

if __name__ == "__main__":
  sys.exit(main())
