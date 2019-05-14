import xnmt.networks as models

class AutoRegressiveTranslator(networks.ConditionedModel, networks.GeneratorModel):
  def calc_nll(self, src, trg):  raise NotImplementedError()
  def initial_state(self, src): raise NotImplementedError()
  def add_input(self, inp, state): raise NotImplementedError()
  def finish_generating(self, output, dec_state): raise NotImplementedError()

import xnmt.networks.translators.seq2seq
import xnmt.networks.translators.ensemble
import xnmt.networks.translators.transformer

