import argparse
import itertools

parser = argparse.ArgumentParser()
parser.add_argument("prob_file", type=argparse.FileType("r"))
parser.add_argument("text_file", type=argparse.FileType("r"))
args = parser.parse_args()

for prob_line, text_line in itertools.zip_longest(args.prob_file, args.text_file):
  assert not(prob_line is None or text_line is None), "Line numbers are not the same."
  p = prob_line.strip().split()
  t = text_line.strip().split() + ["</s>"]
  
  assert len(p) == len(t), "Unmatching length of prob and text"
  
  max_col = [max([len(x), len(y)]) for x, y in zip(p, t)]
  
  line1 = []
  for text, mc in zip(t, max_col):
    line1.append("{}{}".format(text, " " * (mc-len(text))))
  
  line2 = []
  for prob, mc in zip(p, max_col):
    line2.append("{}{}".format(prob, " " * (mc-len(prob))))
  
  print(" ".join(line1))
  print(" ".join(line2))
  print()
  
