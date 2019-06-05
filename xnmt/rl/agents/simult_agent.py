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
  
  def position(self):
    return self.num_writes

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
                                                            lambda: nn.Linear(3 * default_layer_dim,
                                                                                 default_layer_dim))
    self.policy_network = self.add_serializable_component(
      "policy_network", policy_network,
      lambda: xnmt.rl.TransformPolicyNetwork(
        transform=nn.NonLinear(
          input_dim=default_layer_dim,
          output_dim=default_layer_dim
        ),
        scorer=nn.Softmax(
          input_dim=default_layer_dim,
          vocab_size=xnmt.structs.vocabs.SimultActionVocab.VOCAB_SIZE,
          softmax_mask=[0,1,2,3,6,7,8,9]
        )
      )
    )
    self.default_layer_dim = default_layer_dim

  def initial_state(self, src: xnmt.Batch) -> models.PolicyAgentState:
    policy_state = self.policy_network.initial_state(src) if self.policy_network is not None else None
    return models.PolicyAgentState(src, policy_state)

  def next_action(self, state: SimultSeqLenUniDirectionalState) -> Tuple[models.SearchAction, models.PolicyAgentState]:
    if (xnmt.is_train() and self.oracle_in_train) or (not xnmt.is_train() and self.oracle_in_test) or state.force_oracle:
      oracle_action = np.array([state.oracle_batch[i][state.timestep] for i in range(state.oracle_batch.batch_size())])
    else:
      oracle_action = None
    # Training policy?
    if self.policy_network is not None:
      network_state = self.add_input_to_network(state)
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

    return models.SearchAction(state.decoder_state, np.asarray(new_actions, dtype=int), log_likelihood, policy_action.log_softmax, policy_action.mask)

  def add_input_to_network(self, state: SimultSeqLenUniDirectionalState) -> models.PolicyAgentState:
    zeros = lambda: dy.zeros(self.default_layer_dim, batch_size=state.src.batch_size())
    encoder_state = dy.nobackprop(state.encoder_state()) if state.timestep > 0 else zeros()
    decoder_state = dy.nobackprop(state.decoder_state.context()) if state.decoder_state is not None  else zeros()
    trg_embedding = dy.nobackprop(state.decoder_state.prev_embedding if state.decoder_state is not None and \
                                                                        state.decoder_state.prev_embedding is not None \
                                    else zeros())
    network_input = dy.concatenate([encoder_state, decoder_state, trg_embedding])

    return state.network_state.add_input(self.input_transform.transform(network_input))


  def calc_loss(self, dec_state: models.UniDirectionalState, ref: xnmt.Batch):
    return self.policy_network.calc_loss(dec_state, ref)

  def finish_generating(self, state: SimultSeqLenUniDirectionalState):
    return all([x == y for x, y in zip(state.num_writes, state.trg_counts)])


class SimultPolicyAttentionAgent(SimultPolicyAgent, xnmt.Serializable):
  yaml_tag = "!SimultPolicyAttentionAgent"

  @xnmt.serializable_init
  def __init__(self,
               input_transform: Optional[models.Transform] = None,
               policy_network: Optional[networks.PolicyNetwork] = None,
               oracle_in_train: bool = False,
               oracle_in_test: bool = False,
               default_layer_dim: int = xnmt.default_layer_dim,
               encoder_attender: models.Attender = xnmt.bare(nn.MlpAttender),
               decoder_attender: models.Attender = xnmt.bare(nn.MlpAttender)):
    super().__init__(input_transform, policy_network, oracle_in_train, oracle_in_test, default_layer_dim)
    self.encoder_attender = encoder_attender
    self.decoder_attender = decoder_attender

  def initial_state(self, src: xnmt.Batch):
    return models.DoubleAttentionPolicyAgentState(
      src=src,
      policy_state=self.policy_network.initial_state(src) if self.policy_network is not None else None,
      encoder_state=self.encoder_attender.initial_state(),
      decoder_state=self.decoder_attender.initial_state()
    )

  def add_input_to_network(self, state: SimultSeqLenUniDirectionalState) -> models.DoubleAttentionPolicyAgentState:
    zeros = lambda: dy.zeros(self.default_layer_dim, batch_size=state.src.batch_size())
    encoder_state = dy.nobackprop(state.encoder_state()) if state.timestep > 0 else zeros()
    decoder_state = dy.nobackprop(state.decoder_state.context()) if state.decoder_state is not None else zeros()
    trg_embedding = dy.nobackprop(state.decoder_state.prev_embedding if state.decoder_state is not None and \
                                                                        state.decoder_state.prev_embedding is not None \
                                    else zeros())
    
    if isinstance(state.network_state, models.DoubleAttentionPolicyAgentState):
      if state.parent is None:
        read_flag = state.num_reads
        write_flag = state.num_writes
      else:
        read_flag = state.num_reads - state.parent.num_reads
        write_flag = state.num_writes - state.parent.num_writes
      read_flag = np.expand_dims(read_flag, axis=1)
      write_flag = np.expand_dims(write_flag, axis=1)

      # Appending new item in the sentences
      new_encoder_state = self.concat_attender_state(
        prev_state=state.network_state.encoder_state,
        next_state=self.encoder_attender.initial_state(xnmt.ExpressionSequence([encoder_state])),
        mask=xnmt.Mask(1-read_flag)
      )
      new_decoder_state = self.concat_attender_state(
        prev_state=state.network_state.decoder_state,
        next_state=self.decoder_attender.initial_state(xnmt.ExpressionSequence([decoder_state])),
        mask=xnmt.Mask(1-write_flag)
      )
    else:
      raise ValueError()

    query_v = state.network_state.policy_state.output() if state.parent is not None else zeros()
    enc_context, new_encoder_state = self.encoder_attender.calc_context(query_v, new_encoder_state)
    dec_context, new_decoder_state = self.encoder_attender.calc_context(query_v, new_decoder_state)

    network_input = dy.concatenate([enc_context+encoder_state, dec_context+decoder_state, trg_embedding])
    return models.DoubleAttentionPolicyAgentState(
      src = state.network_state.src,
      policy_state=state.network_state.policy_state.add_input(self.input_transform.transform(network_input)),
      encoder_state=new_encoder_state,
      decoder_state=new_decoder_state
    )

  def concat_attender_state(self, prev_state: models.AttenderState, next_state: models.AttenderState, mask: xnmt.Mask) \
      -> models.AttenderState:
    if prev_state.curr_sent is None:
      sent = dy.concatenate([next_state.curr_sent], d=1)
      context = dy.concatenate([next_state.sent_context], d=1)
    else:
      sent = dy.concatenate([prev_state.curr_sent, next_state.curr_sent], d=1)
      context = dy.concatenate([prev_state.sent_context, next_state.sent_context], d=1)

    if prev_state.input_mask is not None:
      mask = xnmt.Mask(np.concatenate([prev_state.input_mask.np_arr, mask.np_arr], axis=1))
    return models.AttenderState(curr_sent=sent, sent_context=context, input_mask=mask)
