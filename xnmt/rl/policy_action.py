
import dynet as dy

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

  def __repr__(self):
    ll = dy.exp(self.log_likelihood).npvalue() if self.log_likelihood is not None else None
    return "({}, {})".format(repr(self.content), ll)
