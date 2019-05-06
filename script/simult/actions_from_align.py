import argparse

parser = argparse.ArgumentParser()
parser.add_argument("align")
parser.add_argument("src_file")
parser.add_argument("trg_file")
parser.add_argument("--debug", action='store_true')
args = parser.parse_args()

def read_data():
  """
  Reading alignment file, ex line:
  1-3 2-4 5-3
  """
  with open(args.align) as align_fp, \
       open(args.src_file) as src_fp, \
       open(args.trg_file) as trg_fp:
    for src_line, trg_line, align in zip(src_fp, trg_fp, align_fp):
      len_src = len(src_line.strip().split())
      len_trg = len(trg_line.strip().split())
      align = align.strip().split()
      align = [x.split("-") for x in align]
      align = [(int(f), int(e)) for f, e in align]
      yield len_src, len_trg, align

def split_alignment(align):
  f_to_e = {}
  e_to_f = {}
  for f, e in align:
    if f not in f_to_e: f_to_e[f] = []
    if e not in e_to_f: e_to_f[e] = []
    f_to_e[f].append(e)
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

def action_from_align(len_src, len_trg, align):
  align = align_missings(len_src, len_trg, align)
  f_to_e, e_to_f = split_alignment(align)
  actions = []

  f_cover = -1
  for j in range(len_trg):
    max_f_cover = max(e_to_f[j])
    if f_cover < max_f_cover:
      actions.extend(["READ"] * (max_f_cover - f_cover))
      f_cover = max_f_cover
    actions.append("WRITE")
  if f_cover+1 != len_src:
    actions.extend(["READ"] * (len_src - f_cover - 1))
  assert len(actions) == (len_src + len_trg)

  # Check before return
  return actions
  
def main():
  for data in read_data():
    actions = action_from_align(*data)
    print(" ".join(actions))

if __name__ == '__main__':
  main()
