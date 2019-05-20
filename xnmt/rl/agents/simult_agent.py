import dynet as dy
import xnmt
import xnmt.models as models
import xnmt.modules.nn as nn
import xnmt.rl.policy_networks as networks

from typing import Optional, Tuple, List


class SimultSeqLenUniDirectionalState(models.UniDirectionalState):
  def __init__(self,
               src: xnmt.Batch,
               full_encodings: xnmt.ExpressionSequence,
               decoder_state: Optional[models.UniDirectionalState] = None,
               num_reads: int = 0,
               num_writes: int = 0,
               simult_action: Optional[models.SearchAction] = None,
               reset_attender: bool = False,
               network_state: Optional[models.PolicyAgentState] = None,
               parent: Optional['SimultSeqLenUniDirectionalState'] = None,
               force_oracle: bool = True):
    self.decoder_state = decoder_state
    self.num_reads = num_reads
    self.num_writes = num_writes
    self.simult_action = simult_action
    self.reset_attender = reset_attender
    self.full_encodings = full_encodings
    self.parent = parent
    self.src = src
    self.network_state = network_state
    self.force_oracle = force_oracle

  def output(self) -> dy.Expression:
    return self.decoder_state.output()
  
  def context(self) -> dy.Expression:
    return self.decoder_state.context()
  
  def add_input(self, word: dy.Expression, mask: Optional[xnmt.Mask] = None):
    raise NotImplementedError("Should not call this line of code")
  
  def encoder_state(self) -> dy.Expression:
    return self.full_encodings[self.num_reads-1] if self.num_reads > 0 else None
  
  def num_actions(self):
    return self.num_reads + self.num_writes
 
  def encodings_to_now(self) -> models.EncoderState:
    encodings = xnmt.ExpressionSequence(self.full_encodings[:self.num_reads])
    return models.EncoderState(encodings, None)

  def collect_trajectories_backward(self) -> List['SimultSeqLenUniDirectionalState']:
    now = self
    ret = []
    while now.parent is not None:
      ret.append(now)
      now = now.parent
    return ret


class SimultPolicyAgent(xnmt.models.PolicyAgent, xnmt.Serializable):
  yaml_tag = "!SimultPolicyAgent"
  READ = 0
  WRITE = 1
  
  __ACTIONS__ = [READ, WRITE]
  
  @xnmt.serializable_init
  def __init__(self,
               input_transform: Optional[models.Transform] = None,
               policy_network: Optional[networks.PolicyNetwork] = None,
               oracle_in_train: bool = False,
               oracle_in_test: bool = False,
               trivial_read_before_write: bool = False,
               trivial_exchange_read_write: bool = False,
               default_layer_dim: int = xnmt.default_layer_dim):
    self.input_transform = input_transform
    self.oracle_in_train = oracle_in_train
    self.oracle_in_test = oracle_in_test
    self.trivial_read_before_write = trivial_read_before_write
    self.trivial_exchange_read_write = trivial_exchange_read_write
    self.input_transform = self.add_serializable_component("input_transform", input_transform,
                                                            lambda: nn.NonLinear(2 * default_layer_dim,
                                                                                 default_layer_dim))
    self.policy_network = self.add_serializable_component("policy_network", policy_network,
                                                          lambda: xnmt.rl.TransformPolicyNetwork(
                                                            nn.Softmax(input_dim=default_layer_dim,
                                                                       vocab_size=len(self.__ACTIONS__))))
    self.default_layer_dim = default_layer_dim
    
    if self.policy_network is None and not trivial_exchange_read_write and not trivial_read_before_write:
      xnmt.logger.info("Policy network is not found for SimultPolicyNetwork, setting up trivia to read before write")
      self.trivial_read_before_write = True
  
  def initial_state(self, src: xnmt.Batch) -> models.PolicyAgentState:
    assert src.batch_size() == 1
    policy_state = self.policy_network.initial_state(src) if self.policy_network is not None else None
    return models.PolicyAgentState(src, policy_state)
  
  def next_action(self, state: Optional[SimultSeqLenUniDirectionalState] = None) \
      -> Tuple[models.SearchAction, models.PolicyAgentState]:
    timestep = state.num_reads + state.num_writes
    oracle_action = getattr(state.src[0], "oracle", None)
    # Define oracle
    if self.trivial_read_before_write:
      oracle_action = self.READ if state.num_reads < state.src.len_unpadded() else self.WRITE
    elif self.trivial_exchange_read_write:
      oracle_action = self.READ if state.num_reads <= state.num_writes else self.WRITE
    elif (xnmt.is_train() and self.oracle_in_train) or (not xnmt.is_train() and self.oracle_in_test):
      oracle_action = oracle_action[timestep]
    else:
      oracle_action = None
    # Training policy?
    if self.policy_network is not None:
      input_state = self.input_transform.transform(self.input_state(state))
      network_state = state.network_state.add_input(input_state)
      
      if xnmt.is_train():
        policy_action = self.policy_network.sample(network_state, 1)[0]
      else:
        policy_action = self.policy_network.best_k(network_state, 1)[0]
      
      # Overriding decision
      if oracle_action is not None and oracle_action != policy_action.action_id:
        policy_action = models.SearchAction(policy_action.decoder_state,
                                            oracle_action, dy.pick(policy_action.log_softmax, oracle_action),
                                            policy_action.log_softmax, policy_action.mask)
      
    else:
      policy_action = models.SearchAction(action_id=oracle_action)
      network_state = state.network_state
    
    return policy_action, network_state
  
  def input_state(self, state: SimultSeqLenUniDirectionalState):
    encoder_state = state.encoder_state() or dy.zeros(self.default_layer_dim)
    decoder_state = state.decoder_state.context() if state.decoder_state is not None else dy.zeros(self.default_layer_dim)
    return dy.concatenate([dy.nobackprop(encoder_state), dy.nobackprop(decoder_state)])

    
  def calc_loss(self, dec_state: models.UniDirectionalState, ref: xnmt.Batch, cached_softmax: Optional[dy.Expression] = None):
    return self.policy_network.calc_loss(dec_state, ref, cached_softmax)
  
  def finish_generating(self, state: SimultSeqLenUniDirectionalState):
    oracle_action = getattr(state.src[0], "oracle")
    return state.num_reads + state.num_writes < len(oracle_action)
  

