import argparse

argparse = argparse.ArgumentParser()
argparse.add_argument("f_sp_input")
argparse.add_argument("e_sp_input")
argparse.add_argument("align_file")
args = argparse.parse_args()

DELIMITER = "▁"

def main():
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
    
    # Mapping piece
    new_align = []
    for f, e in align:
      for _, fpiece in f_sp_word[f]:
        for _, epiece in e_sp_word[e]:
          new_align.append((fpiece, epiece))
    new_align = sorted(new_align)
    print(" ".join(["{}-{}".format(f, e) for f, e in new_align]))

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

if __name__ == '__main__':
  main()
