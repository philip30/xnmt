import dynet as dy
import numpy as np
import xnmt
import xnmt.models as models
import xnmt.modules.nn as nn
import xnmt.rl.policy_networks as networks

from typing import Optional, Tuple, Iterator, List


class SimultSeqLenUniDirectionalState(models.UniDirectionalState):
  def __init__(self,
               src: xnmt.Batch,
               full_encodings: xnmt.ExpressionSequence,
               oracle_batch: xnmt.Batch = None,
               trg_counts: Optional[List[int]] = None,
               decoder_state: Optional[nn.decoders.arb_len.ArbSeqLenUniDirectionalState] = None,
               num_reads: Optional[List[int]] = None,
               num_writes: Optional[List[int]] = None,
               simult_action: Optional[models.SearchAction] = None,
               read_was_performed: bool = False,
               network_state: Optional[models.PolicyAgentState] = None,
               parent: Optional['SimultSeqLenUniDirectionalState'] = None,
               force_oracle: bool = False,
               timestep: int = 0):
    self.decoder_state = decoder_state
    self.num_reads = num_reads if num_reads is not None else np.zeros(src.batch_size(), dtype=int)
    self.num_writes = num_writes if num_writes is not None else np.zeros(src.batch_size(), dtype=int)
    self.oracle_batch = oracle_batch
    self.simult_action = simult_action
    self.read_was_performed = read_was_performed
    self.full_encodings = full_encodings
    self.parent = parent
    self.src = src
    self.network_state = network_state
    self.force_oracle = force_oracle
    self.timestep = timestep
    self.trg_counts = trg_counts

  def output(self) -> dy.Expression:
    return self.decoder_state.output()

  def context(self) -> dy.Expression:
    return self.decoder_state.context()

  def add_input(self, word: dy.Expression, mask: Optional[xnmt.Mask] = None):
    raise NotImplementedError("Should not call this line of code")

  def encoder_state(self) -> dy.Expression:
    mask = np.zeros((len(self.full_encodings), len(self.num_reads)))
    for i, read in enumerate(self.num_reads):
      if read != 0:
        mask[read-1, i] = 1
    mask = dy.inputTensor(mask, batched=True)
    return self.full_encodings.as_tensor() * mask

  def num_actions(self):
    return self.timestep

  def collect_trajectories_backward(self) -> Iterator['SimultSeqLenUniDirectionalState']:
    now = self
    while now.parent is not None:
      yield now
      now = now.parent


class SimultPolicyAgent(xnmt.models.PolicyAgent, xnmt.Serializable):
  yaml_tag = "!SimultPolicyAgent"
  READ = xnmt.structs.vocabs.SimultActionVocab.READ
  WRITE = xnmt.structs.vocabs.SimultActionVocab.WRITE

  __ACTIONS__ = [READ, WRITE]

  @xnmt.serializable_init
  def __init__(self,
               input_transform: Optional[models.Transform] = None,
               policy_network: Optional[networks.PolicyNetwork] = None,
               oracle_in_train: bool = False,
               oracle_in_test: bool = False,
               default_layer_dim: int = xnmt.default_layer_dim):
    self.input_transform = input_transform
    self.oracle_in_train = oracle_in_train
    self.oracle_in_test = oracle_in_test

    self.input_transform = self.add_serializable_component("input_transform", input_transform,
                                                            lambda: nn.NonLinear(2 * default_layer_dim,
                                                                                 default_layer_dim))
    self.policy_network = self.add_serializable_component("policy_network", policy_network,
                                                          lambda: xnmt.rl.TransformPolicyNetwork(
                                                            nn.Softmax(input_dim=default_layer_dim,
                                                                       vocab_size=xnmt.structs.vocabs.SimultActionVocab.VOCAB_SIZE)))
    self.default_layer_dim = default_layer_dim

  def initial_state(self, src: xnmt.Batch) -> models.PolicyAgentState:
#    assert src.batch_size() == 1
    policy_state = self.policy_network.initial_state(src) if self.policy_network is not None else None
    return models.PolicyAgentState(src, policy_state)

  def next_action(self, state: SimultSeqLenUniDirectionalState) \
      -> Tuple[models.SearchAction, models.PolicyAgentState]:
    if (xnmt.is_train() and self.oracle_in_train) or (not xnmt.is_train() and self.oracle_in_test) or state.force_oracle:
      oracle_action = np.array([state.oracle_batch[i][state.timestep] for i in range(state.oracle_batch.batch_size())])
    else:
      oracle_action = None
    # Training policy?
    if self.policy_network is not None:
      input_state = self.input_transform.transform(self.input_state(state))
      network_state = state.network_state.add_input(input_state)
      if oracle_action is None:
        if xnmt.is_train():
          policy_action = self.policy_network.sample(network_state, 1)[0]
        else:
          policy_action = self.policy_network.best_k(network_state, 1)[0]
      else:
        policy_action = self.policy_network.pick_oracle(oracle_action, network_state)[0]
    else:
      policy_action = models.SearchAction(action_id=oracle_action)
      network_state = state.network_state

    if oracle_action is None:
      policy_action = self.check_sanity(state, policy_action)

    return policy_action, network_state

  def check_sanity(self, state: SimultSeqLenUniDirectionalState, policy_action: models.SearchAction):
    src_len  = np.array([state.src[i].len_unpadded() for i in range(state.src.batch_size())])
    num_reads = state.num_reads
    actions = policy_action.action_id

    new_actions = []
    modified =  False
    for l, r, a in zip(src_len, num_reads, actions):
      if l == r and a == self.READ:
        new_actions.append(self.WRITE)
        modified = True
      else:
        new_actions.append(a)

    if modified and policy_action.log_likelihood is not None:
      log_likelihood = dy.pick_batch(policy_action.log_softmax, actions)
    else:
      log_likelihood = policy_action.log_softmax

    return models.SearchAction(state.decoder_state, new_actions, log_likelihood, policy_action.log_softmax, policy_action.mask)

  def input_state(self, state: SimultSeqLenUniDirectionalState):
    encoder_state = state.encoder_state() if state.timestep > 0 \
                    else dy.zeros(self.default_layer_dim, batch_size=state.src.batch_size())
    decoder_state = state.decoder_state.context() if state.decoder_state is not None \
                    else dy.zeros(self.default_layer_dim, batch_size=state.src.batch_size())
    return dy.concatenate([dy.nobackprop(encoder_state), dy.nobackprop(decoder_state)])

  def calc_loss(self, dec_state: models.UniDirectionalState, ref: xnmt.Batch):
    return self.policy_network.calc_loss(dec_state, ref)

  def finish_generating(self, state: SimultSeqLenUniDirectionalState):
    return all([x == y for x, y in zip(state.num_writes, state.trg_counts)])


