import dynet as dy
import numbers

import xnmt.transducers.base as transducers_base
import xnmt.modelparts.decoders as decoders
import xnmt.transducers.recurrent as recurrent
import xnmt.expression_seqs as expr_seq
import xnmt.vocabs as vocabs

from xnmt.rl.policy_action import PolicyAction

class SimultaneousState(decoders.AutoRegressiveDecoderState):
  """
  The read/write state used to determine the state of the SimultaneousTranslator.
  """
  def __init__(self,
               model,
               encoder_state: recurrent.UniLSTMState,
               decoder_state: decoders.AutoRegressiveDecoderState,
               has_been_read:int = 0,
               has_been_written:int = 0,
               src_encoding: dy.Expression = None,
               written_word: numbers.Integral = None,
               policy_action: PolicyAction=None,
               reset_attender:bool = True,
               parent: 'SimultaneousState' = None):
    super().__init__(None, None)
    self.model = model
    self.encoder_state = encoder_state
    self.decoder_state = decoder_state
    self.has_been_read = has_been_read
    self.has_been_written = has_been_written
    self.written_word = written_word
    self.policy_action = policy_action
    self.reset_attender = reset_attender
    self.src_encoding = src_encoding
    self.cache = {}
    self.parent = parent
    
  def read(self, src, policy_action):
    src_embed = self.model.src_embedder.embed(src[self.has_been_read])
    next_encoder_state = self.encoder_state.add_input(src_embed)
    
    return SimultaneousState(self.model, next_encoder_state, self.decoder_state,
                             has_been_read=self.has_been_read+1, has_been_written=self.has_been_written,
                             src_encoding=next_encoder_state.output(), written_word=self.written_word,
                             policy_action=policy_action, reset_attender=True,
                             parent = self)
  
  def write(self, word, policy_action):
    # Reset attender if there is a read action
    reset_attender = self.reset_attender
    if reset_attender:
      encodings = self.find_backward("src_encoding")
      self.model.attender.init_sent(expr_seq.ReversedExpressionSequence(expr_list=encodings))
      reset_attender = False

    # Generating h_t based on RNN(h_{t-1}, embed(e_{t-1}))
    if self.decoder_state is None or word is None:
      fin_tran_state = [transducers_base.FinalTransducerState(h, c) for h, c in \
                        zip(self.encoder_state.h(), self.encoder_state.c())]
      decoder_state = self.model.decoder.initial_state(fin_tran_state, vocabs.Vocab.SS)
    else:
      decoder_state = self.model.decoder.add_input(self.decoder_state, word)
    decoder_state.attention = self.model.attender.calc_attention(decoder_state.as_vector())
    decoder_state.context = self.model.attender.calc_context(None, decoder_state.attention)
    
    # Calc context for decoding
    return SimultaneousState(self.model, self.encoder_state, decoder_state,
                             has_been_read=self.has_been_read, has_been_written=self.has_been_written+1,
                             src_encoding=None, written_word=word, policy_action=policy_action,
                             reset_attender=reset_attender, parent=self)
  
  def find_backward(self, field):
    now = self
    results = []
    while now.parent is not None:
      if field in now.cache:
        if len(results) == 0:
          results = now.cache[field]
        else:
          results.extend(now.cache[field])
        break
      else:
        result = getattr(now, field)
        if result is not None:
          results.append(result)
      now = now.parent
    self.cache[field] = results
    return self.cache[field]
 
  # These states are used for decoding
  @property
  def rnn_state(self):
    return self.decoder_state.rnn_state
  
  @property
  def context(self):
    return self.decoder_state.context
  
  def __repr__(self):
    content = self.policy_action.content if self.policy_action is not None else None
    return "({}, {}, {})".format(content, self.has_been_read, self.has_been_written)

