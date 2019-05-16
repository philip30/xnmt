
import xnmt
import xnmt.models as models
import xnmt.modules.nn as nn



class PolicyNetwork(models.Policy, xnmt.Serializable):
  yaml_tag = "!PolicyNetwork"
  @xnmt.serializable_init
  def __init__(self, policy_network: models.Transform = xnmt.bare(nn.Linear)):
    self.policy_network = policy_network

  def input_state(self, state):
    return self.policy_network.transform(state)


class RecurrentPolicyNetwork(models.Policy, nn.Linear):
  @xnmt.serializable_init
  def __init__(self,
               policy_network: models.Transform = xnmt.bare(nn.Linear),
               rnn: models.SeqTransducer = xnmt.bare(nn.BiLSTMSeqTransducer)):
    self.rnn = rnn
    self.policy_network = policy_network

  ### Warning do not use single sample_action normally here!
  ### Please use sample_actions to sample for making sequetial decisions

  def sample_actions(self, states: xnmt.ExpressionSequence, argmax=False, sample_pp=None, predefined_action=None):
    states = self.rnn.transduce(states)
    return super().sample_actions(states.encode_seq, argmax, sample_pp, predefined_action)

  def input_state(self, state):
    return self.policy_network.transform(state)
