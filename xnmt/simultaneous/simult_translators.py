import dynet as dy

from enum import Enum
from collections import defaultdict

import xnmt.input_readers as input_readers
import xnmt.modelparts.embedders as embedders
import xnmt.modelparts.attenders as attenders
import xnmt.modelparts.decoders as decoders
import xnmt.inferences as inferences
import xnmt.transducers.recurrent as recurrent
import xnmt.events as events
import xnmt.event_trigger as event_trigger
import xnmt.vocabs as vocabs
import xnmt.sent as sent
import xnmt.losses as losses

from xnmt.models.base import PolicyConditionedModel
from xnmt.models.translators.default import DefaultTranslator
from xnmt.persistence import bare, Serializable, serializable_init
from xnmt.rl.policy_network import PolicyNetwork
from xnmt.rl.policy_action import PolicyAction

from .simult_state import SimultaneousState

class SimultaneousTranslator(DefaultTranslator, PolicyConditionedModel, Serializable):
  yaml_tag = '!SimultaneousTranslator'
  
  class Action(Enum):
    READ = 0
    WRITE = 1
  
  @serializable_init
  def __init__(self,
               src_reader: input_readers.InputReader,
               trg_reader: input_readers.InputReader,
               src_embedder: embedders.Embedder = bare(embedders.SimpleWordEmbedder),
               encoder: recurrent.UniLSTMSeqTransducer = bare(recurrent.UniLSTMSeqTransducer),
               attender: attenders.Attender = bare(attenders.MlpAttender),
               decoder: decoders.Decoder = bare(decoders.AutoRegressiveDecoder),
               inference: inferences.AutoRegressiveInference = bare(inferences.AutoRegressiveInference),
               truncate_dec_batches: bool = False,
               policy_network: PolicyNetwork = None,
               max_generation=100,
               policy_train_oracle=False,
               policy_test_oracle=False,
               policy_sample=False,
               read_before_write=False,
               logger=None) -> None:
    super().__init__(src_reader=src_reader,
                     trg_reader=trg_reader,
                     encoder=encoder,
                     attender=attender,
                     src_embedder=src_embedder,
                     decoder=decoder,
                     inference=inference,
                     truncate_dec_batches=truncate_dec_batches)
    PolicyConditionedModel.__init__(self)
    self.max_generation = max_generation
    self.logger = logger
    self.policy_train_oracle = policy_train_oracle
    self.policy_test_oracle = policy_test_oracle
    self.policy_sample = policy_sample
    self.policy_network = policy_network
    self.read_before_write = read_before_write
    
  def create_trajectories(self, src_batch, trg_batch):
    if len(self.actions) != 0:
      return
    
    from_oracle = self.policy_train_oracle if self.train else self.policy_test_oracle
    
    for src, trg in zip(src_batch, trg_batch):
      actions, outputs, decoder_states, model_states = \
        self.create_trajectory(src, trg, from_oracle=from_oracle, force_decoding=True)
      self.actions.append(actions)
      self.outputs.append(outputs)
      self.decoder_states.append(decoder_states)
      self.model_states.append(model_states)
    
  def calc_nll(self, src_batch, trg_batch) -> losses.LossExpr:
    event_trigger.start_sent(src_batch)
    self.create_trajectories(src_batch, trg_batch)
    
    batch_loss = []
    for src, trg, decoder_state in zip(src_batch, trg_batch, self.decoder_states):
      seq_loss = [self.decoder.calc_loss(decoder_state[i], trg[i]) for i in range(len(decoder_state))]
      batch_loss.append(dy.esum(seq_loss))
    
    dy.forward(batch_loss)
    total_loss =  dy.concatenate_to_batch(batch_loss)
    total_units = [trg_batch[i].len_unpadded() for i in range(trg_batch.batch_size())]
    return losses.LossExpr(total_loss, total_units)
  
  def calc_policy_nll(self, src_batch, trg_batch) -> losses.LossExpr:
    assert self.policy_network is not None
  
    event_trigger.start_sent(src_batch)
    self.create_trajectories(src_batch, trg_batch)
  
    batch_loss = []
    for src, action, model_states in zip(src_batch, self.actions, self.model_states):
      assert type(src) == sent.CompoundSentence
      policy_actions = reversed(model_states[-1].find_backward("policy_action"))
      ref_action = src.sents[1].words
      seq_ll = [dy.pick(act.log_likelihood, ref) for act, ref in zip(policy_actions, ref_action)]
      batch_loss.append(-dy.esum(seq_ll))

    dy.forward(batch_loss)
    total_loss = dy.concatenate_to_batch(batch_loss)
    total_units = [len(x) for x in self.actions]
    return losses.LossExpr(total_loss, total_units)
    
    
  def add_input(self, prev_word, state) -> DefaultTranslator.Output:
    src = self.src[0]
    if type(src) == sent.CompoundSentence:
      src_sent = src.sents[0]
    else:
      src_sent = src
    if type(prev_word) == list:
      prev_word = prev_word[0]
      
    look_oracle = self.policy_train_oracle if self.train else self.policy_test_oracle
    # Reading until next write
    while state.has_been_read < src.sent_len():
      if look_oracle:
        force_action = src.sents[1][state.has_been_read + state.has_been_written]
      else:
        force_action = None
      next_action = self._next_action(state, src.sent_len(), force_action)
      if next_action == self.Action.WRITE.value:
        break
      else:
        state = state.read(src_sent)
    # Write one output without reference
    state = state.write(prev_word)
  
    return DefaultTranslator.Output(state, state.decoder_state.attention)
  
  def _initial_state(self, src):
    encoder_state = self.encoder.initial_state()
    return SimultaneousState(self, encoder_state=encoder_state, decoder_state=None)

  def create_trajectory(self, src, ref=None, current_state=None, from_oracle=True, force_decoding=True):
    if type(src) == sent.CompoundSentence:
      src, force_action = src.sents[0], src.sents[1]
    else:
      force_action = defaultdict(lambda: None)
   
    if not from_oracle:
      force_action = defaultdict(lambda: None)
      
    current_state = current_state or self._initial_state(src)
    src_len = src.sent_len()

    actions = []
    outputs = []
    decoder_states = []
    model_states = [current_state]

    def stoping_criterions_met(state, trg, now_action):
      look_oracle = self.policy_train_oracle if self.train else self.policy_test_oracle
      if look_oracle:
        return state.has_been_written + state.has_been_read >= len(now_action.words)
      elif self.policy_network is None:
        return state.has_been_written >= trg.sent_len()
      else:
        return state.has_been_written >= self.max_generation or \
               state.prev_written_word == vocabs.Vocab.ES

    # Simultaneous greedy search
    while not stoping_criterions_met(current_state, ref, force_action):
      # Define action based on state
      policy_action = self._next_action(current_state, src_len, force_action[len(actions)])
      action = policy_action.content
      if action == self.Action.READ.value:
        # Reading + Encoding
        current_state = current_state.read(src, policy_action)
      elif action == self.Action.WRITE.value:
        # Calculating losses
        if force_decoding:
          if ref.sent_len() <= current_state.has_been_written:
            ref_word = vocabs.Vocab.ES
          else:
            ref_word = ref[current_state.has_been_written]
        else:
          ref_word = None
        # Write
        current_state = current_state.write(ref_word, policy_action)
        # The produced words
        decoder_states.append(current_state)
        outputs.append(current_state.written_word)
      else:
        raise ValueError(action)
        
      model_states.append(current_state)
      actions.append(action)
      
    return actions, outputs, decoder_states, model_states

  def _next_action(self, state, src_len, force_action=None) -> PolicyAction:
    nopolicy_nooracle = force_action is None and self.policy_network is None
    if self.read_before_write or nopolicy_nooracle:
      if state.has_been_read < src_len:
        force_action = self.Action.READ.value
      else:
        force_action = self.Action.WRITE.value
    
    # Sanity Check here:
    if force_action is None:
      force_action = self.Action.READ.value if state.has_been_read == 0 else force_action # No writing at the beginning.
      force_action = self.Action.WRITE.value if state.has_been_read == src_len else force_action # No reading at the end.
    
    # Compose inputs from 3 states
    if self.policy_network is not None:
      encoder_state = state.encoder_state.output()
      enc_dim = encoder_state.dim()
      decoder_state = state.decoder_state.as_vector() if state.decoder_state is not None else dy.zeros(*enc_dim)
      policy_input = dy.nobackprop(dy.concatenate([encoder_state, decoder_state]))
      predefined_action = [force_action] if force_action is not None else None
      # Sample / Calculate a single action
      policy_action = self.policy_network.sample_action(policy_input,
                                                        predefined_actions=predefined_action,
                                                        argmax=not (self.train and self.policy_sample))
      policy_action.single_action()
    else:
      policy_action = PolicyAction(force_action)
      
    return policy_action

  @events.handle_xnmt_event
  def on_set_train(self, train):
    self.train = train
    
  @events.handle_xnmt_event
  def on_start_sent(self, src_batch):
    index = src_batch[0].idx
    
    if not hasattr(self, "last_index") or index != self.last_index:
      self.src = src_batch
      self.actions = []
      self.outputs = []
      self.decoder_states = []
      self.model_states = []
      self.last_index = index
    
#  @events.handle_xnmt_event
#  def on_calc_reinforce_loss(self, trg, generator, generator_loss):
#    if self.policy_learning is None:
#      return None
#    reward, bleu, delay, instant_rewards = rewards.SimultaneousReward(self.src, trg, self.actions, self.outputs, self.trg_reader.vocab).calculate()
#    results = {}
#    reinforce_loss = self.policy_learning.calc_loss(reward, results)
#    try:
#      return reinforce_loss
#    finally:
#      if self.logger is not None:
#        keywords = {
#          "sim_inputs": [x[:x.len_unpadded()+1] for x in self.src],
#          "sim_actions": self.actions,
#          "sim_outputs": self.outputs,
#          "sim_bleu": bleu,
#          "sim_delay": delay,
#          "sim_instant_reward": instant_rewards,
#        }
#        keywords.update(results)
#        self.logger.create_sent_report(**keywords)
#
