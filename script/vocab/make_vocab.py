#!/usr/bin/env python3

"""
Simple script to generate vocabulary that can be used in most of the xnmt.input_readers

--min_count Is a filter based on count of words that need to be at least min_count to appear in the vocab.

"""


import sys
import argparse
import math
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("--min_count", type=int, default=1)
parser.add_argument("--top", type=int, default=-1)
parser.add_argument("--prob", type=float, default=-1)
args = parser.parse_args()

all_words = Counter()
for line in sys.stdin:
  all_words.update(line.strip().split())

if args.min_count > 1:
  all_words = [(key, value) for key, value in all_words.items() if value >= args.min_count]
else:
  all_words = list(all_words.items())

all_words = list(map(list, all_words))

all_words = sorted(all_words, key=lambda x: -x[1])

if args.top != -1 and args.top < len(all_words):
  all_words = all_words[:args.top]

if args.prob >= 0.0:
  sum_word = sum([x[1] for x in all_words])
  for i in range(len(all_words)):
    all_words[i][1] = math.log(all_words[i][1]/sum_word)


for word, value in sorted(all_words, key=lambda x: x[1], reverse=True):
  print("{}\t{}".format(word, value))

