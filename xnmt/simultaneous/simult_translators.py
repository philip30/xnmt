import dynet as dy

from enum import Enum
from typing import Optional

import xnmt.batchers as batchers
import xnmt.input_readers as input_readers
import xnmt.modelparts.embedders as embedders
import xnmt.modelparts.attenders as attenders
import xnmt.modelparts.decoders as decoders
import xnmt.inferences as inferences
import xnmt.transducers.recurrent as recurrent
import xnmt.event_trigger as event_trigger
import xnmt.vocabs as vocabs
import xnmt.sent as sent
import xnmt.losses as losses

from xnmt import logger
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
               src_embedder: embedders.Embedder = bare(embedders.LookupEmbedder),
               encoder: recurrent.UniLSTMSeqTransducer = bare(recurrent.UniLSTMSeqTransducer),
               attender: attenders.Attender = bare(attenders.MlpAttender),
               decoder: decoders.Decoder = bare(decoders.AutoRegressiveDecoder),
               inference: inferences.AutoRegressiveInference = bare(inferences.AutoRegressiveInference),
               truncate_dec_batches: bool = False,
               policy_network: Optional[PolicyNetwork] = None,
               policy_train_oracle=False,
               policy_test_oracle=False,
               policy_sample=False,
               read_before_write=False) -> None:
    super().__init__(src_reader=src_reader,
                     trg_reader=trg_reader,
                     encoder=encoder,
                     attender=attender,
                     src_embedder=src_embedder,
                     decoder=decoder,
                     inference=inference,
                     truncate_dec_batches=truncate_dec_batches)
    policy_network = self.add_serializable_component("policy_network", policy_network, lambda: policy_network)
    PolicyConditionedModel.__init__(self, policy_network, policy_train_oracle, policy_test_oracle)
    self.policy_sample = policy_sample
    self.read_before_write = read_before_write

    if self.read_before_write:
      logger.info("Setting looking oracle to always false in SimultTranslator for 'read_before_write'")
      self.policy_train_oracle = False
      self.policy_test_oracle = False

    self.outputs = []
    self.decoder_states = []
    self.model_states = []

  def _is_action_forced(self):
    return self.read_before_write

  def reset_policy_states(self):
    super().reset_policy_states()
    self.outputs.clear()
    self.decoder_states.clear()
    self.model_states.clear()

  def calc_nll(self, src_batch, trg_batch) -> losses.LossExpr:
    event_trigger.start_sent(src_batch)
    self.create_trajectories(src_batch, trg_batch, force_oracle=not self._is_action_forced())

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
    self.create_trajectories(src_batch, trg_batch, force_oracle=not self._is_action_forced())

    batch_loss = []
    for src, action, model_states in zip(src_batch, self.actions, self.model_states):
      policy_actions = model_states[-1].find_backward("policy_action")
      seq_ll = [dy.pick(act.log_likelihood, act.content) for act in policy_actions]
      batch_loss.append(-dy.esum(seq_ll))

    dy.forward(batch_loss)
    total_loss = dy.concatenate_to_batch(batch_loss)
    total_units = [len(x) for x in self.actions]
    return losses.LossExpr(total_loss, total_units)

  def add_input(self, prev_word, state) -> DefaultTranslator.Output:
    if batchers.is_batched(self.src_sents):
      src = self.src_sents[0]
    else:
      src = self.src_sents

    force_actions = None
    look_oracle = self.policy_train_oracle if self.train else self.policy_test_oracle
    if type(src) == sent.CompoundSentence:
      src, force_actions = src.sents[0], src.sents[1].words
    force_actions = force_actions if look_oracle else None
    if type(prev_word) == list:
      prev_word = prev_word[0]

    while True:
      force_action = None

      # If we look at the oracle, fill the value of force action accordingly
      if look_oracle:
        now_position = state.has_been_read + state.has_been_written
        # Case when in the inference the produced sentence is longer than the reference.
        # The number of enough reads have been taken, just need to write the output until the end.
        if now_position < len(force_actions):
          force_action = force_actions[now_position]
        else:
          force_action = self.Action.WRITE.value

      # Taking the next action
      next_action = self._next_action(state, src.len_unpadded(), force_action)

      if next_action.content == self.Action.WRITE.value:
        state = state.write(self.src_encoding, prev_word, next_action)
        break
      elif next_action.content == self.Action.READ.value:
        state = state.read(self.src_encoding[state.has_been_read], next_action)
      else:
        raise ValueError(next_action.content)

    return DefaultTranslator.Output(state, state.decoder_state.attention)

  def _initial_state(self, src):
    if batchers.is_batched(src):
      src = src[0]
    if type(src) == sent.CompoundSentence:
      src = src.sents[0]
    self.src_encoding = self.encoder.transduce(self.src_embedder.embed_sent(src))
    return SimultaneousState(self, encoder_state=None, decoder_state=None)

  def create_trajectories(self, src_batch, trg_batch, force_oracle=False, parent_model=None, **kwargs):
    if len(self.actions) != 0:
      return

    from_oracle = self.policy_train_oracle if self.train else self.policy_test_oracle
    from_oracle = from_oracle or force_oracle

    for src, trg in zip(src_batch, trg_batch):
      actions, outputs, decoder_states, model_states = \
        self.create_trajectory(src, trg, from_oracle=from_oracle, force_decoding=True)
      self.actions.append(actions)
      self.outputs.append(outputs)
      self.decoder_states.append(decoder_states)
      self.model_states.append(model_states)

  def create_trajectory(self,
                        src: sent.Sentence,
                        ref: sent.Sentence = None,
                        current_state: Optional[SimultaneousState] = None,
                        from_oracle: bool = True,
                        force_decoding: bool = True,
                        max_generation: int = -1):
    assert not from_oracle or type(src) == sent.CompoundSentence or self._is_action_forced()
    force_action = None
    if type(src) == sent.CompoundSentence:
      src, force_action = src.sents[0], src.sents[1].words
    force_action = force_action if from_oracle else None
    current_state = current_state or self._initial_state(src)
    src_len = src.len_unpadded()

    actions = []
    decoder_states = []
    outputs = []
    model_states = [current_state]

    def stoping_criterions_met(state, trg, now_action):
      look_oracle = now_action is not None and from_oracle
      if look_oracle:
        return state.has_been_read + state.has_been_written >= len(force_action)
      elif self.policy_network is None or self._is_action_forced():
        return state.has_been_written >= trg.sent_len()
      else:
        return (max_generation != -1 and state.has_been_written >= max_generation) or \
               state.written_word == vocabs.Vocab.ES

    # Simultaneous greedy search
    while not stoping_criterions_met(current_state, ref, force_action):
      actions_taken = current_state.has_been_read + current_state.has_been_written
      if force_action is not None and actions_taken < len(force_action):
        defined_action = force_action[actions_taken]
      else:
        defined_action = None

      # Define action based on state
      policy_action = self._next_action(current_state, src_len, defined_action)
      action = policy_action.content

      if action == self.Action.READ.value:
        # Reading + Encoding
        current_state = current_state.read(self.src_encoding[current_state.has_been_read], policy_action)

      elif action == self.Action.WRITE.value:
        # Calculating losses
        if force_decoding:
          if ref.len_unpadded() <= current_state.has_been_written:
            prev_word = vocabs.Vocab.ES
          elif current_state.has_been_written == 0:
            prev_word = None
          else:
            prev_word = ref[current_state.has_been_written-1]
          # Write
          current_state = current_state.write(self.src_encoding, prev_word, policy_action)
        else:
          # TODO implement if ref is None!
          pass
        # The produced words
        outputs.append(prev_word)
        decoder_states.append(current_state)

      else:
        raise ValueError(action)

      model_states.append(current_state)
      actions.append(action)

    return actions, outputs, decoder_states, model_states

  def _next_action(self, state, src_len, force_action=None) -> PolicyAction:
    # Sanity Check here:
    if force_action is None:
      force_action = self.Action.READ.value if state.has_been_read == 0 else force_action # No writing at the beginning.
      force_action = self.Action.WRITE.value if state.has_been_read == src_len else force_action # No reading at the end.

    if self.read_before_write:
      force_action = self.Action.READ.value if state.has_been_read < src_len else self.Action.WRITE.value

    # Compose inputs from 3 states
    if self.policy_network is not None:
      enc_dim = self.src_encoding[0].dim()
      encoder_state = state.encoder_state if state.encoder_state is not None else dy.zeros(*enc_dim)
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

    # TODO(philip30): Update this value when you add more actions
    if policy_action.content > 2:
      import random
      policy_action.content = random.randint(0, 1)

    return policy_action


