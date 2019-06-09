import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("src_file")
parser.add_argument("trg_file")
parser.add_argument("action_file")
parser.add_argument("--add-eos", action='store_true')
args = parser.parse_args()

with open(args.src_file) as src_fp, \
     open(args.trg_file) as trg_fp, \
     open(args.action_file) as action_fp:
  for src_line, trg_line, action_line in zip(src_fp, trg_fp, action_fp):
    src_line = src_line.strip().split()
    trg_line = trg_line.strip().split()

    if args.add_eos:
      src_line.append("</s>")
      trg_line.append("</s>")

    action_fp = action_line.strip().split()

    src_line = iter(src_line)
    trg_line = iter(trg_line)

    last_action = -1
    src_output = []
    trg_output = []
    for a in action_fp:
      if (a == "READ" or a == "PREDICT_READ") and (last_action == "WRITE" or last_action == "PREDICT_WRITE") or\
         (a == "WRITE" or a == "PREDICT_WRITE") and (last_action == "READ" or last_action == "PREDICT_READ"):
        if a == "READ" or a == "PREDICT_READ":
          trg_output.append("|")
        else:
          src_output.append("|")
      if a == "READ":
        if a == "PREDICT_READ":
          src_output.append("[{}]".format(next(src_line)))
        else:
          src_output.append(next(src_line))
      else:
        if a == "PREDICT_WRITE":
          trg_output.append("[{}]".format(next(trg_line)))
        else:
          trg_output.append(next(trg_line))
      last_action = a
    
    print(" ".join(src_output))
    print(" ".join(trg_output))
    print()

