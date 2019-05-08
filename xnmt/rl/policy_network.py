
import dynet as dy

import xnmt.modelparts.transforms as transforms
import xnmt.events as events
import xnmt.transducers.base as seq_transducer
import xnmt.expression_seqs as expr_seq
import xnmt.transducers.recurrent as recurrent

from xnmt.rl.policy_action import PolicyAction
from xnmt.persistence import Serializable, serializable_init, bare


class Policy(object):
  def sample_actions(self, states: expr_seq.ExpressionSequence, argmax=False, sample_pp=None, predefined_action=None):
    actions = []
    for i, (state, predef_act) in enumerate(zip(states, predefined_action)):
      mask = states.mask[i] if states.mask[i] is not None else None
      actions.append(self.sample_action(state, argmax, sample_pp, predefined_action, mask))
    return actions

  def sample_action(self, state, argmax=False, sample_pp=None, predefined_actions=None, mask=None) -> PolicyAction:
    """
    state: Input state.
    argmax: Whether to perform argmax or sampling.
    sample_pp: Stands for sample post_processing.
               Every time the sample are being drawn, this method will be invoked with sample_pp(sample).
    predefined_actions: Whether to forcefully the network to assign the action value to some predefined actions.
                        This predefined actions can be from the gold distribution or some probability priors.
                        It should be calculated from the outside.
    """
    policy = dy.log_softmax(self.input_state(state))

    # Select actions
    if predefined_actions is not None and len(predefined_actions) != 0:
      actions = predefined_actions
    else:
      if argmax:
        actions = policy.tensor_value().argmax().as_numpy()[0]
      else:
        actions = policy.tensor_value().categorical_sample_log_prob().as_numpy()[0]
      if len(actions.shape) == 0:
        actions = [actions]
    # Post Processing
    if sample_pp is not None:
      actions = sample_pp(actions)
    # Return
    return PolicyAction(actions, policy, state, mask)
  
  def input_state(self, state):
    raise NotImplementedError()


class PolicyNetwork(Serializable, Policy):

  yaml_tag = '!PolicyNetwork'
  @serializable_init
  @events.register_xnmt_handler
  def __init__(self, policy_network: transforms.Transform = bare(transforms.Linear)):
    self.policy_network = policy_network

  def input_state(self, state):
    return self.policy_network.transform(state)


class RecurrentPolicyNetwork(Serializable, Policy):

  yaml_tag = '!RecurrentPolicyNetwork'
  @serializable_init
  @events.register_xnmt_handler
  def __init__(self,
               policy_network: transforms.Transform = bare(transforms.Linear),
               rnn: seq_transducer.SeqTransducer = bare(recurrent.BiLSTMSeqTransducer)):
    self.rnn = rnn
    self.policy_network = policy_network
    
  ### Warning do not use single sample_action normally here!
  ### Please use sample_actions to sample for making sequetial decisions

  def sample_actions(self, states: expr_seq.ExpressionSequence, argmax=False, sample_pp=None, predefined_action=None):
    states = self.rnn.transduce(states)
    return super().sample_actions(states, argmax, sample_pp, predefined_action)
    
  def input_state(self, state):
    return self.policy_network.transform(state)
