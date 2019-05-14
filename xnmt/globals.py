### GlobalStates
class GlobalStates(object):
  def __init__(self):
    self.train = False
    self.src_batch = None
    self.SAAM_LOG_ATTENTION = False


singleton_global = None
if singleton_global is None:
  singleton_global = GlobalStates()

### CONSTANTS
INF = 1e20
EPS = 1e-10

### Quick helper method
def is_train():
  return singleton_global.train
