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

    print(" ".join(shift_actions_according_to_prob(actions, sprob, tprob, args.threshold)))



def shift_actions_according_to_prob(actions, sprob, tprob, threshold):
  s_ctr = 0
  t_ctr = 0
  for i in range(len(actions)):
    act = actions[i]
    if act == READ:
      if sprob is not None and sprob[s_ctr] > threshold:
        actions[i] = PR_READ
        j = i-1
        while j >= 0 and actions[j] == WRITE:
          actions[j], actions[j+1] = actions[j+1], actions[j]
          j -= 1
      s_ctr += 1
    elif act == WRITE:
      if tprob is not None and tprob[t_ctr] > threshold:
        actions[i] = PR_WRITE
        j= i-1
        while j >= 0 and actions[j] == READ:
          actions[j], actions[j+1] = actions[j+1], actions[j]
          j -= 1
    else:
      raise ValueError()
  return actions
  


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("actions")
  parser.add_argument("--src_prob")
  parser.add_argument("--trg_prob")
  parser.add_argument("--threshold", default=0.5)
  main(parser.parse_args())
  

