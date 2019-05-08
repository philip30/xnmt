import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("src_file")
parser.add_argument("trg_file")
parser.add_argument("action_file")
args = parser.parse_args()

with open(args.src_file) as src_fp, \
     open(args.trg_file) as trg_fp, \
     open(args.action_file) as action_fp:
  for src_line, trg_line, action_line in zip(src_fp, trg_fp, action_fp):
    src_line = iter(src_line.strip().split())
    trg_line = iter(trg_line.strip().split())
    action_fp = action_line.strip().split()

    last_action = -1
    src_output = []
    trg_output = []
    for a in action_fp:
      if a != last_action and last_action != -1:
        if a == "READ":
          trg_output.append("|")
        else:
          src_output.append("|")
      if a == "READ":
        src_output.append(next(src_line))
      else:
        trg_output.append(next(trg_line))
      last_action = a
    
    print(" ".join(src_output))
    print(" ".join(trg_output))
    print()

