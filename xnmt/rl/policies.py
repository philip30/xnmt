import xnmt.modules.nn.transforms as transforms
import xnmt.internal.events as events
import xnmt.modules.transducers.base as seq_transducer
import xnmt.structs.expression_seqs as expr_seq
import xnmt.modules.transducers.recurrent as recurrent

from xnmt.internal.persistence import Serializable, serializable_init, bare




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
