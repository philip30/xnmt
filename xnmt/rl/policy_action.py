
class PolicyAction(object):
  def __init__(self, content, log_likelihood=None, policy_input=None, mask=None):
    self._content = content
    self._log_likelihood = log_likelihood
    self._mask = mask
    self._policy_input = policy_input

  @property
  def content(self):
    return self._content
  
  @property
  def log_likelihood(self):
    return self._log_likelihood
  
  @property
  def mask(self):
    return self._mask
  
  @property
  def policy_input(self):
    return self._policy_input
  
  def single_action(self):
    self._content = self._content[0]
