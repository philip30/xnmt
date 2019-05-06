import argparse
import sys

argparse = argparse.ArgumentParser()
argparse.add_argument("f_sp_input")
argparse.add_argument("e_sp_input")
argparse.add_argument("align_file")
argparse.add_argument("--debug", action='store_true')
args = argparse.parse_args()

DELIMITER = "▁"

def main():
  exit_status = 0
  for line, (f_sp, e_sp, align) in enumerate(iter_input(args.f_sp_input, args.e_sp_input, args.align_file)):
    # Grouping piece together
    f_sp_word = []
    for i, piece in enumerate(f_sp):
      if DELIMITER in piece:
        f_sp_word.append([])
      f_sp_word[-1].append((piece, i))
    e_sp_word = []
    for i, piece in enumerate(e_sp):
      if DELIMITER in piece:
        e_sp_word.append([])
      e_sp_word[-1].append((piece, i))

    align = align_missings(len(f_sp_word), len(e_sp_word), align)

    if args.debug:
      debug(f_sp_word, e_sp_word, align)

    # Mapping piece
    try:
      new_align = []
      for f, e in align:
        for _, fpiece in f_sp_word[f]:
          for _, epiece in e_sp_word[e]:
            new_align.append((fpiece, epiece))
      new_align = sorted(new_align)
      print(" ".join(["{}-{}".format(f, e) for f, e in new_align]))
    except:
      print("Error on line: {}".format(line+1), file=sys.stderr)
      exit_status = 1
  sys.exit(exit_status)

def iter_input(f_sp_file, e_sp_file, align_file):
  with open(f_sp_file) as f_sp_fp, \
       open(e_sp_file) as e_sp_fp, \
       open(align_file) as align_fp:
    for f_sp_line, e_sp_line, align_line in zip(f_sp_fp, e_sp_fp, align_fp):
      f_sp = f_sp_line.strip().split()
      e_sp = e_sp_line.strip().split()
      align = align_line.strip().split()
      align = [x.split("-") for x in align]
      align = [(int(f), int(e)) for f, e in align]
      yield f_sp, e_sp, align

def split_alignment(align):
  f_to_e = {}
  e_to_f = {}
  for f, e in align:
    if f not in f_to_e:
      f_to_e[f] = []
    f_to_e[f].append(e)
    if e not in e_to_f:
      e_to_f[e] = []
    e_to_f[e].append(f)
  return f_to_e, e_to_f

def align_missings(len_f, len_e, align):
  f_to_e, e_to_f = split_alignment(align)
  missing = []

  for i in range(len_e-1, -1, -1):
    if i not in e_to_f:
      f_align = e_to_f[i+1] if i != len_e-1 else [len_f-1]
      for f in f_align:
        missing.append((f, i))
      e_to_f[i] = f_align
  return align + missing

def debug(f_sp_word, e_sp_word, align):
  f_to_e, e_to_f = split_alignment(align)
  for i in range(len(e_sp_word)):
    for f in e_to_f[i]:
      print("Aligning {} with {}".format(e_sp_word[i], f_sp_word[f]), file=sys.stderr)

  print(file=sys.stderr)
        
if __name__ == '__main__':
  main()
