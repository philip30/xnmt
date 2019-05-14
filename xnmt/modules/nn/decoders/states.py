
import dynet as dy

import xnmt
import xnmt.models as models
import xnmt.structs.sentences as sent


class AutoRegressiveDecoderState(models.DecoderState):
  """A state holding all the information needed for AutoRegressiveDecoder

  Args:
    rnn_state: a DyNet RNN state
    context: a DyNet expression
  """
  def __init__(self,
               rnn_state: models.UniDirectionalState,
               context: dy.Expression,
               attender_state: models.AttenderState):
    self._rnn_state = rnn_state
    self._attender_state = attender_state
    self._context = context

  @property
  def context(self):
    return self._context

  @property
  def attender_state(self):
    return self._attender_state

  @property
  def rnn_state(self):
    return self._rnn_state

  def as_vector(self):
    return self._rnn_state.output()


class RNNGDecoderState(models.DecoderState):
  """A state holding all the information needed for RNNGDecoder

  Args:
    stack
    context
  """
  def __init__(self, stack, context, word_read=0, num_open_nt=0, finish_generating=False, initial_state=None):
    self._stack = stack
    self._context = context
    self._word_read = word_read
    self._num_open_nt = num_open_nt
    self._finish_generating = finish_generating

  # DecoderState interface
  def as_vector(self): return self.stack[-1].output()
  # Public accessible fields
  @property
  def context(self): return self._context
  @context.setter
  def context(self, value): self._context = value
  @property
  def stack(self): return self._stack
  @property
  def word_read(self): return self._word_read
  @property
  def num_open_nt(self): return self._num_open_nt
  @property
  def finish_generating(self): return self._finish_generating


  def gen(self, word_encoding, finish_generating, shift_from_enc):
    h_i = self.stack[-1].add_input(word_encoding)
    stack_i = [x for x in self.stack] + [self.RNNGStackState(h_i, sent.RNNGAction.Type.NONE)]
    inc_read = 1 if shift_from_enc else 0
    return RNNGDecoderState(stack=stack_i,
                            context=self.context,
                            word_read=self.word_read+inc_read,
                            num_open_nt=self.num_open_nt,
                            finish_generating=finish_generating)

  def reduce(self, is_left, edge_id, head_composer, edge_embedder):
    children = self.stack[-2:]
    if is_left: children = reversed(children)
    children = [child.output() for child in children]
    edge_embedding = edge_embedder.embed(xnmt.mark_as_batch([edge_id]))
    children.append(edge_embedding)
    x_i = head_composer.compose(children)
    h_i = self.stack[-3].add_input(x_i)
    stack_i = self.stack[:-2] + [self.RNNGStackState(h_i, sent.RNNGAction.Type.NONE)]
    return RNNGDecoderState(stack=stack_i,
                            context=self.context,
                            word_read=self.word_read,
                            num_open_nt=self.num_open_nt,
                            finish_generating=self.finish_generating)

  def nt(self, nt_embed):
    h_i = self.stack[-1].add_input(nt_embed)
    stack_i = [x for x in self.stack] + [self.RNNGStackState(h_i, sent.RNNGAction.Type.NT)]
    return RNNGDecoderState(stack=stack_i,
                            context=self.context,
                            word_read=self.word_read,
                            num_open_nt=self.num_open_nt+1,
                            finish_generating=self.finish_generating)

  def reduce_nt(self, nt_embedder, head_composer):
    num_pop = 0
    while self.stack[-(num_pop+1)].action != sent.RNNGAction.Type.REDUCE_NT:
      num_pop += 1
    children = self.stack[-num_pop:]
    children = [child.output() for child in children]
    head_embedding = nt_embedder.embed(self.stack[-(num_pop+1)].action.action_content)
    children.append(head_embedding)
    x_i = head_composer.transduce(children)
    h_i = self.stack[-(num_pop+1)].add_input(x_i)
    stack_i = self.stack[:-num_pop] + [self.RNNGStackState(h_i)]
    return RNNGDecoderState(stack=stack_i,
                            context=self.context,
                            word_read=self.word_read,
                            num_open_nt=self.num_open_nt-1,
                            finish_generating=self.finish_generating)

  class RNNGStackState(object):
    def __init__(self, stack_content, stack_action):
      self._content = stack_content
      self._action = stack_action

    def add_input(self, x):
      return RNNGDecoderState.RNNGStackState(self._content.add_input(x), self._action)

    def output(self):
      return self._content.output()

    @property
    def action(self):
      return self._action


class SimultaneousState(AutoRegressiveDecoderState):
  """
  The read/write state used to determine the state of the SimultaneousTranslator.
  """
  def __init__(self,
               model,
               encoder_state: models.UniDirectionalState,
               decoder_state: models.DecoderState,
               has_been_read:int = 0,
               has_been_written:int = 0,
               written_word: int = None,
               policy_action: models.SearchAction = None,
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
    self.cache = {}
    self.parent = parent

  def read(self, src_encoding, policy_action):
    return SimultaneousState(self.model, src_encoding, self.decoder_state,
                             has_been_read=self.has_been_read+1, has_been_written=self.has_been_written,
                             written_word=self.written_word, policy_action=policy_action, reset_attender=True,
                             parent=self)

  def write(self, src_encoding, word, policy_action):
    # Reset attender if there is a read action
    reset_attender = self.reset_attender
    if reset_attender:
      encodings = src_encoding[:self.has_been_read]
      self.model.attender.init_sent(xnmt.ExpressionSequence(expr_list=encodings))
      reset_attender = False

    # Generating h_t based on RNN(h_{t-1}, embed(e_{t-1}))
    if self.decoder_state is None or word is None:
      dim = src_encoding[0].dim()
      fin_tran_state = [models.FinalTransducerState(dy.zeros(*dim), dy.zeros(*dim))]
      decoder_state = self.model.decoder.initial_state(fin_tran_state, xnmt.Vocab.SS)
    else:
      decoder_state = self.model.decoder.add_input(self.decoder_state, word)
    decoder_state.attention = self.model.attender.calc_attention(decoder_state.as_vector())
    decoder_state.context = self.model.attender.calc_context(decoder_state.as_vector(), decoder_state.attention)

    # Calc context for decoding
    return SimultaneousState(self.model, self.encoder_state, decoder_state,
                             has_been_read=self.has_been_read, has_been_written=self.has_been_written+1,
                             written_word=word, policy_action=policy_action, reset_attender=reset_attender,
                             parent=self)

  def find_backward(self, field) -> list:
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

