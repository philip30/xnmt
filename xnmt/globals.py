### GlobalStates
class GlobalStates(object):
  def __init__(self):
    self.train = False
    self.reporting = False
    self.SAAM_LOG_ATTENTION = False


singleton_global = None
if singleton_global is None:
  singleton_global = GlobalStates()

### CONSTANTS
INF = 1e20
EPS = 1e-10
NO_DECODING_ATTEMPTED = "@@NO_DECODING_ATTEMPTED@@"

### Quick helper method
def is_train():
  return singleton_global.train

def is_reporting():
  return singleton_global.reporting
