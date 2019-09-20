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
  PREDICT_READ = xnmt.structs.vocabs.SimultActionVocab.PREDICT_READ
  PREDICT_WRITE = xnmt.structs.vocabs.SimultActionVocab.PREDICT_WRITE

  __ACTIONS__ = [READ, WRITE, PREDICT_READ, PREDICT_WRITE]

  @xnmt.serializable_init
  def __init__(self,
               input_transform: Optional[models.Transform] = None,
               policy_network: Optional[networks.PolicyNetwork] = None,
               oracle_in_train: bool = False,
               oracle_in_test: bool = False,
               default_layer_dim: int = xnmt.default_layer_dim,
               dropout: Optional[float] = 0.0):
    self.input_transform = input_transform
    self.oracle_in_train = oracle_in_train
    self.oracle_in_test = oracle_in_test
    self.dropout = dropout
    self.input_transform = self.add_serializable_component("input_transform", input_transform,
                                                            lambda: nn.Linear(3 * default_layer_dim,
                                                                              default_layer_dim))
    self.policy_network = self.add_serializable_component(
      "policy_network", policy_network,
      lambda: xnmt.rl.RecurrentPolicyNetwork(
        rnn=nn.UniLSTMSeqTransducer(input_dim=default_layer_dim, hidden_dim=default_layer_dim),
        transform=nn.NonLinear(
          input_dim=default_layer_dim,
          output_dim=default_layer_dim
        ),
        scorer=nn.Softmax(
          input_dim=default_layer_dim,
          vocab_size=xnmt.structs.vocabs.SimultActionVocab.VOCAB_SIZE,
          softmax_mask=[0,1,2,3,8,9]
        )
      )
    )
    self.default_layer_dim = default_layer_dim

  def initial_state(self, src: xnmt.Batch) -> models.PolicyAgentState:
    policy_state = self.policy_network.initial_state(src) if self.policy_network is not None else None
    return models.PolicyAgentState(src, policy_state)

  def next_action(self, state: SimultSeqLenUniDirectionalState, force_oracle = False, is_sample = False) -> Tuple[models.SearchAction, models.PolicyAgentState]:
    if (xnmt.is_train() and self.oracle_in_train) or (not xnmt.is_train() and self.oracle_in_test) or force_oracle:
      oracle_action = np.array([state.oracle_batch[i][state.timestep] if state.timestep < state.oracle_batch.sent_len() \
                                  else SimultPolicyAgent.WRITE
                                for i in range(state.oracle_batch.batch_size())])
    else:
      oracle_action = None
    # Training policy?
    if self.policy_network is not None:
      network_state = self.add_input_to_network(state)
      if oracle_action is None:
        if is_sample:
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
      if l == r and (a == self.READ or a == self.PREDICT_READ):
        new_actions.append(self.WRITE)
        modified = True
      else:
        new_actions.append(a)

    if modified and policy_action.log_likelihood is not None:
      log_likelihood = dy.pick_batch(policy_action.log_softmax, actions)
    else:
      log_likelihood = policy_action.log_likelihood

    return models.SearchAction(state.decoder_state, np.asarray(new_actions, dtype=int), log_likelihood, policy_action.log_softmax, policy_action.mask)

  def add_input_to_network(self, state: SimultSeqLenUniDirectionalState) -> models.PolicyAgentState:
    zeros = lambda: dy.zeros(self.default_layer_dim, batch_size=state.src.batch_size())
    estate = state.encoder_state()if state.timestep > 0 else zeros()
    dstate = state.decoder_state
    encoder_state = dy.nobackprop(estate)
    decoder_state = dy.nobackprop(dstate.merged_context) if dstate.merged_context is not None else dstate.rnn_state.output()
    trg_embedding = dy.nobackprop(dstate.prev_embedding) if dstate.prev_embedding is not None else zeros()
    if xnmt.is_train() and self.dropout > 0.0:
      encoder_state = dy.dropout(encoder_state, self.dropout)
      decoder_state = dy.dropout(decoder_state, self.dropout)
      trg_embedding = dy.dropout(trg_embedding, self.dropout)

    network_input = dy.concatenate([encoder_state, decoder_state, trg_embedding])
    return state.network_state.add_input(self.input_transform.transform(network_input))


  def calc_loss(self, dec_state: models.UniDirectionalState, ref: xnmt.Batch):
    return self.policy_network.calc_loss(dec_state, ref)

  def finish_generating(self, state: SimultSeqLenUniDirectionalState):
    return state.timestep == state.oracle_batch.sent_len()


class SimultPolicyAttentionAgent(SimultPolicyAgent, xnmt.Serializable):
  yaml_tag = "!SimultPolicyAttentionAgent"

  @xnmt.serializable_init
  def __init__(self,
               input_transform: Optional[models.Transform] = None,
               policy_network: Optional[networks.PolicyNetwork] = None,
               oracle_in_train: bool = False,
               oracle_in_test: bool = False,
               default_layer_dim: int = xnmt.default_layer_dim,
               encoder_attender: models.Attender = xnmt.bare(nn.DotAttender),
               decoder_attender: models.Attender = xnmt.bare(nn.DotAttender),
               encoder_q_transform: models.Transform = xnmt.bare(nn.Linear),
               encoder_k_transform: models.Transform = xnmt.bare(nn.Linear),
               encoder_v_transform: models.Transform = xnmt.bare(nn.Linear),
               decoder_q_transform: models.Transform = xnmt.bare(nn.Linear),
               decoder_k_transform: models.Transform = xnmt.bare(nn.Linear),
               decoder_v_transform: models.Transform = xnmt.bare(nn.Linear)):
    super().__init__(input_transform, policy_network, oracle_in_train, oracle_in_test, default_layer_dim)
    self.encoder_attender = encoder_attender
    self.decoder_attender = decoder_attender
    self.encoder_q_transform = encoder_q_transform
    self.encoder_k_transform = encoder_k_transform
    self.encoder_v_transform = encoder_v_transform
    self.decoder_q_transform = decoder_q_transform
    self.decoder_k_transform = decoder_k_transform
    self.decoder_v_transform = decoder_v_transform

  def initial_state(self, src: xnmt.Batch):
    return models.DoubleAttentionPolicyAgentState(
      src=src,
      policy_state=self.policy_network.initial_state(src) if self.policy_network is not None else None,
      encoder_state=self.encoder_attender.initial_state(),
      decoder_state=self.decoder_attender.initial_state()
    )

  def add_input_to_network(self, state: SimultSeqLenUniDirectionalState) -> models.DoubleAttentionPolicyAgentState:
    zeros = lambda: dy.constant(self.default_layer_dim, val=1e-3, batch_size=state.src.batch_size())
    estate = state.encoder_state() if state.timestep > 0 else zeros()
    dstate = state.decoder_state
    encoder_state = dy.nobackprop(estate)
    decoder_state = dy.nobackprop(dstate.merged_context) if dstate.merged_context is not None else zeros()
    trg_embedding = dy.nobackprop(dstate.prev_embedding) if dstate.prev_embedding is not None else zeros()

    if isinstance(state.network_state, models.DoubleAttentionPolicyAgentState):
      if state.parent is None:
        read_flag = state.num_reads
        write_flag = state.num_writes
      else:
        read_flag = state.num_reads - state.parent.num_reads
        write_flag = state.num_writes - state.parent.num_writes
      read_flag = np.expand_dims(read_flag, axis=1)
      write_flag = np.expand_dims(write_flag, axis=1)


      encoder_key = encoder_state
      encoder_value = encoder_state
      decoder_key = decoder_state
      decoder_value = decoder_state
      # Appending new item in the sentences
      new_encoder_state = self.concat_attender_state(
        prev_state=state.network_state.encoder_state,
        next_state=self.encoder_attender.initial_state(xnmt.ExpressionSequence([encoder_key]),
                                                       xnmt.ExpressionSequence([encoder_value])),
        mask=xnmt.Mask(1-read_flag)
      )
      new_decoder_state = self.concat_attender_state(
        prev_state=state.network_state.decoder_state,
        next_state=self.decoder_attender.initial_state(xnmt.ExpressionSequence([decoder_key]),
                                                       xnmt.ExpressionSequence([decoder_value])),
        mask=xnmt.Mask(1-write_flag)
      )
    else:
      raise ValueError()

    query_v = state.network_state.policy_state.output() if state.parent is not None else zeros()
    query_e = self.encoder_q_transform.transform(query_v)
    query_d = self.decoder_q_transform.transform(query_v)
    enc_context, new_encoder_state = self.encoder_attender.calc_context(query_e, new_encoder_state)
    dec_context, new_decoder_state = self.encoder_attender.calc_context(query_d, new_decoder_state)

    network_input = dy.concatenate([encoder_state+enc_context, decoder_state+dec_context, trg_embedding])
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
      value = dy.concatenate([next_state.curr_value], d=1)
      context = dy.concatenate([next_state.sent_context], d=1)
    else:
      sent = dy.concatenate([prev_state.curr_sent, next_state.curr_sent], d=1)
      context = dy.concatenate([prev_state.sent_context, next_state.sent_context], d=1)
      value = dy.concatenate([prev_state.curr_value, next_state.curr_value], d=1)

    if prev_state.input_mask is not None:
      mask = xnmt.Mask(np.concatenate([prev_state.input_mask.np_arr, mask.np_arr], axis=1))
    return models.AttenderState(curr_sent=sent, sent_context=context, input_mask=mask, curr_value=value)
