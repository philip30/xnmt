#!/usr/bin/env python3

"""
By: Philip Arthur

Script to generate CHARAGRAM vocabulary.
For example if we have 2 words corpus: ["ab", "deac"]
Then it wil produce the char n gram vocabulary.

a
b
c
d
e
ab
de
ea
ac
dea
eac
deac

This is useful to be used in CharNGramSegmentComposer.

Args:
  ngram - The size of the ngram.
  top - Prin only the top ngram.
"""


import sys
import argparse
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("--ngram", type=int, default=4)
parser.add_argument("--top", type=int, default=-1)
args = parser.parse_args()

def to_ngram(word, ngram):
  counts = Counter()
  for i in range(len(word)):
    for j in range(i+1, min(i+ngram+1, len(word)+1)):
      counts[word[i:j]] += 1
  return counts

counts = Counter()
for line in sys.stdin:
  counts.update(line.strip().split())
   
ngrams = Counter()
for word, count in counts.items():
  word_vect = to_ngram(word, args.ngram)
  word_vect = Counter({k: count*v for k, v in word_vect.items()})
  ngrams.update(word_vect)

for i, (key, count) in enumerate(sorted(ngrams.items(), key=lambda x: -x[1])):
  if args.top != -1:
    if i == args.top:
      break
  print(key)


