
import dynet as dy

import xnmt.modelparts.transforms as transforms
import xnmt.events as events

from xnmt.rl.policy_action import PolicyAction
from xnmt.persistence import Serializable, serializable_init


class PolicyNetwork(Serializable):

  yaml_tag = '!PolicyNetwork'
  @serializable_init
  @events.register_xnmt_handler
  def __init__(self, policy_network: transforms.Transform = None):
    assert policy_network is not None
    self.policy_network = policy_network

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
    policy = dy.log_softmax(self.policy_network.transform(state))

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


