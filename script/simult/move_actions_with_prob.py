import argparse

READ = "READ"
WRITE = "WRITE"
PR_READ = "PREDICT_READ"
PR_WRITE = "PREDICT_WRITE"

def main(args):
  src_prob = None
  trg_prob = None
  
  with open(args.actions) as fp:
    actions = map(lambda x: x.strip().split(), fp.readlines())
  
  parse_prob = lambda line: list(map(float, line.strip().split()))
  if args.src_prob:
    with open(args.src_prob) as fp:
      src_prob = [parse_prob(x) for x in fp.readlines()]
      
  if args.trg_prob:
    with open(args.trg_prob) as fp:
      trg_prob = [parse_prob(x) for x in fp.readlines()]
  
  for i, actions in enumerate(actions):
    sprob = src_prob[i] if src_prob else None
    tprob = trg_prob[i] if trg_prob else None
    ret_actions = shift_actions_according_to_prob(actions, sprob, tprob, args.threshold)
    assert sanity_check(actions, ret_actions)
    print(" ".join(ret_actions))


def count_read_write(act):
  read = 0
  write = 0
  for a in act:
    if a == "READ" or a == "PREDICT_READ": read += 1
    elif a == "WRITE" or a == "PREDICT_WRITE": write += 1
    else: raise ValueError()
  return read, write


def sanity_check(act, ret):
  r1, w1 = count_read_write(act)
  r2, w2 = count_read_write(ret)
  return r1 == r2 and w1 == w2


def shift_actions_according_to_prob(actions, sprob, tprob, threshold):
  s_ctr = 0
  t_ctr = 0
  for i in range(len(actions)):
    act = actions[i]
    if act == READ:
      if sprob is not None and sprob[s_ctr] > threshold:
        j = i-1
        if j >= 0 and actions[j] == WRITE:
          actions[i] = PR_READ
          while j >= 0 and actions[j] == WRITE:
            actions[j], actions[j+1] = actions[j+1], actions[j]
            j -= 1
      s_ctr += 1
    elif act == WRITE:
      if tprob is not None and tprob[t_ctr] > threshold:
        j= i-1
        if j >= 0 and actions[j] == READ:
          actions[i] = PR_WRITE
          while j >= 0 and actions[j] == READ:
            actions[j], actions[j+1] = actions[j+1], actions[j]
            j -= 1
      t_ctr += 1
    else:
      raise ValueError()
  return actions
  


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("actions")
  parser.add_argument("--src_prob")
  parser.add_argument("--trg_prob")
  parser.add_argument("--threshold", type=float, default=0.5)
  main(parser.parse_args())
  

