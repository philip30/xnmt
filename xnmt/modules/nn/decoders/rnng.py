import numpy as np
import dynet as dy
from typing import List

import xnmt
import xnmt.models as models
import xnmt.modules.nn as nn
import xnmt.modules.input_readers as input_readers
import xnmt.structs.sentences as sent

from .states import RNNGDecoderState

# TODO Fix this
class RNNGDecoder(models.Decoder, xnmt.Serializable):
  RNNG_ACTION_SIZE = 6

  @xnmt.serializable_init
  def __init__(self,
               input_dim: int = xnmt.Ref("exp_global.default_layer_dim"),
               head_composer: models.SequenceComposer = xnmt.bare(nn.DyerHeadComposer),
               rnn: models.UniDiSeqTransducer = xnmt.bare(nn.UniLSTMSeqTransducer),
               bridge: models.Bridge = xnmt.bare(nn.NoBridge),
               nt_embedder: models.Embedder = None,
               edge_embedder: models.Embedder = None,
               term_embedder: models.Embedder = None,
               action_scorer: models.Scorer = None,
               nt_scorer: models.Scorer = None,
               term_scorer: models.Scorer = None,
               edge_scorer: models.Scorer = None,
               transform: models.Transform = xnmt.bare(nn.AuxNonLinear),
               ban_actions: List[int] = [1, 4],
               shift_from_enc: bool = True,
               max_open_nt: int = 100,
               graph_reader: input_readers.GraphReader = xnmt.ref_trg_reader):
    self.input_dim = input_dim
    self.rnn = rnn
    self.bridge = bridge
    self.head_composer = head_composer
    self.nt_embedder = self.add_serializable_component("nt_embedder", nt_embedder,
                                                       lambda: nn.LookupEmbedder(
                                                         emb_dim=self.input_dim,
                                                         vocab_size=len(graph_reader.node_vocab)
                                                       ))
    self.edge_embedder = self.add_serializable_component("edge_embedder", edge_embedder,
                                                         lambda: nn.LookupEmbedder(
                                                           emb_dim=self.input_dim,
                                                           vocab_size=len(graph_reader.edge_vocab)
                                                         ))
    self.term_embedder = self.add_serializable_component("term_embedder", term_embedder,
                                                         lambda: nn.LookupEmbedder(
                                                           emb_dim=self.input_dim,
                                                           vocab_size=len(graph_reader.value_vocab)
                                                         ))
    self.transform = self.add_serializable_component("transform", transform, lambda: transform)
    self.action_scorer = self.add_serializable_component("action_scorer", action_scorer,
                                                         lambda: nn.Softmax(
                                                         input_dim=input_dim,
                                                         vocab_size=RNNGDecoder.RNNG_ACTION_SIZE))
    self.nt_scorer = self.add_serializable_component("nt_scorer", nt_scorer,
                                                     lambda: nn.Softmax(
                                                       input_dim=input_dim,
                                                       vocab_size=len(graph_reader.node_vocab)
                                                     ))
    self.term_scorer = self.add_serializable_component("term_scorer", term_scorer,
                                                       lambda: nn.Softmax(
                                                         input_dim=input_dim,
                                                         vocab_size=len(graph_reader.value_vocab)
                                                       ))
    self.edge_scorer = self.add_serializable_component("edge_scorer", edge_scorer,
                                                       lambda: nn.Softmax(
                                                         input_dim=input_dim,
                                                         vocab_size=len(graph_reader.edge_vocab)
                                                       ))
    self.ban_actions = ban_actions
    self.max_open_nt = max_open_nt
    self.shift_from_enc = shift_from_enc

  ### Decoder Interface
  def initial_state(self, enc_final_states, ss_expr):
    rnn_state = self.rnn.initial_state()
    rnn_s = self.bridge.decoder_init(enc_final_states)
    rnn_state = rnn_state.set_s(rnn_s)
    # This is important
    assert ss_expr.batch_size() == 1, "Currently, RNNG could not handle batch size > 1 in training and testing.\n" \
                                      "Please consider using autobatching."
    return RNNGDecoderState(stack=[RNNGDecoderState.RNNGStackState(rnn_state)], context=None)

  def add_input(self, dec_state: RNNGDecoderState, actions: List[sent.RNNGAction]):
    action = actions[0] if xnmt.is_batched(actions) else actions
    action_type = action.action_type
    if action_type == sent.RNNGAction.Type.GEN:
      # Shifting the embedding of a word
      if self.shift_from_enc:
        # Feed in the decoder based on input string
        return self._perform_gen(dec_state, self.sent_enc[dec_state.word_read])
      else:
        # Feed in the decoder based on the previously generated output / oracle output
        return self._perform_gen(dec_state, self.term_embedder.embed(action.action_content),
                                 finish_generating=action.action_content == xnmt.Vocab.ES)
    elif action_type == sent.RNNGAction.Type.REDUCE_LEFT or \
         action_type == sent.RNNGAction.Type.REDUCE_RIGHT:
      # Perform Reduce on Left direction or right direction
      return dec_state.reduce(action == sent.RNNGAction.Type.REDUCE_LEFT,
                              action.action_content, self.head_composer,
                              self.edge_embedder)
    elif action_type == sent.RNNGAction.Type.NT:
      # Shifting the embedding of the NT's head
      return dec_state.nt(action.action_content)
    elif action_type == sent.RNNGAction.Type.REDUCE_NT:
      return dec_state.reduce_nt(self.nt_embedder, self.head_composer)
    elif action_type == sent.RNNGAction.Type.NONE:
      return dec_state
    else:
      raise  NotImplementedError("Unimplemented for action word:", action)

  def calc_loss(self, dec_state, ref_action):
    state = self._calc_transform(dec_state)
    action_batch = xnmt.mark_as_batch([x.action_type.value for x in ref_action])
    action_type = ref_action[0].action_type
    loss = self.action_scorer.calc_loss(state, action_batch)
    # Aux Losses based on action content
    if action_type == sent.RNNGAction.Type.NT:
      nt_batch = xnmt.mark_as_batch([x.action_content for x in ref_action])
      loss += self.nt_scorer.calc_loss(state, nt_batch)
    elif action_type == sent.RNNGAction.Type.GEN:
      term_batch = xnmt.mark_as_batch([x.action_content for x in ref_action])
      loss += self.term_scorer.calc_loss(state, term_batch)
    elif action_type == sent.RNNGAction.Type.REDUCE_LEFT or \
         action_type == sent.RNNGAction.Type.REDUCE_RIGHT:
      edge_batch = xnmt.mark_as_batch([x.action_content for x in ref_action])
      loss += self.edge_scorer.calc_loss(state, edge_batch)
    # Total Loss
    return loss

  def best_k(self, dec_state, k, normalize_scores=False):
    final_state = self._calc_transform(dec_state)
    # p(a)
    action_logprob = self.action_scorer.calc_log_probs(final_state).npvalue()
    # p(nt|a == 'NT')
    action_logprob = np.array([action_logprob[i] for i in range(self.RNNG_ACTION_SIZE)])
    # RULING OUT INVALID ACTIONS
    rule_out = set(self.ban_actions)
    rule_out.add(sent.RNNGAction.Type.NONE.value)
    if len(dec_state.stack) <= 2:
      rule_out.add(sent.RNNGAction.Type.REDUCE_LEFT.value)
      rule_out.add(sent.RNNGAction.Type.REDUCE_RIGHT.value)
    if self.shift_from_enc:
      if dec_state.word_read >= len(self.sent_enc) :
        rule_out.add(sent.RNNGAction.Type.GEN.value)
    else:
      if dec_state.finish_generating:
        rule_out.add(sent.RNNGAction.Type.GEN.value)
    if dec_state.num_open_nt == 0:
      rule_out.add(sent.RNNGAction.Type.REDUCE_NT.value)
    if dec_state.num_open_nt > self.max_open_nt:
      rule_out.add(sent.RNNGAction.Type.NT.value)
    if len(rule_out) == len(action_logprob):
      rule_out.remove(sent.RNNGAction.Type.NONE.value)
    # Nulling out probability
    for action_value in rule_out:
      action_logprob[action_value] = -np.inf
    # Take out best action
    action_type = sent.RNNGAction.Type(np.argmax(action_logprob))
    best_score = action_logprob[action_type.value]
    if action_type == sent.RNNGAction.Type.NT:
      nt_logprob = self.nt_scorer.calc_log_probs(final_state).npvalue()
      return self._find_best_k(action_type, nt_logprob, k, best_score)
    elif action_type == sent.RNNGAction.Type.GEN:
      term_logprob = self.term_scorer.calc_log_probs(final_state).npvalue()
      return self._find_best_k(action_type, term_logprob, k, best_score)
    elif action_type == sent.RNNGAction.Type.REDUCE_LEFT or \
         action_type == sent.RNNGAction.Type.REDUCE_RIGHT:
      edge_logprob = self.edge_scorer.calc_log_probs(final_state).npvalue()
      return self._find_best_k(action_type, edge_logprob, k, best_score)
    else:
      best_action = sent.RNNGAction(action_type)
      return [best_action], [best_score]

  def _find_best_k(self, action_type, logprob, k, action_cond_prob):
    best_k = logprob.argsort()[max(-k, -len(logprob)+1):][::-1]
    actions = []
    scores = []
    for item in best_k:
      actions.append(sent.RNNGAction(action_type=action_type, action_content=item))
      scores.append(action_cond_prob + logprob[item])
    return actions, scores

  def sample(self, dec_state, n, temperature=1.0):
    raise NotImplementedError("Implement this function!")

  def init_sent(self, sent_enc):
    self.sent_enc = sent_enc

  def shared_params(self):
    return [{".embedder.emb_dim", ".rnn.input_dim"},
            {".input_dim", ".rnn.decoder_input_dim"},
            {".input_dim", ".transform.input_dim"},
            {".input_feeding", ".rnn.decoder_input_feeding"},
            {".rnn.layers", ".bridge.dec_layers"},
            {".rnn.hidden_dim", ".bridge.dec_dim"},
            {".rnn.hidden_dim", ".transform.aux_input_dim"},
            {".transform.output_dim", ".scorer.input_dim"}]

  ### RNNGDecoder Modules
  def _calc_transform(self, dec_state):
    return self.transform.transform(dy.concatenate([dec_state.as_vector(), dec_state.context]))

  def finish_generating(self, dec_output, dec_state):
    if type(dec_output) == np.ndarray or type(dec_output) == list:
      assert len(dec_output) == 1
    gen_finish = dec_state.word_read == len(self.sent_enc) if self.shift_from_enc else dec_state.finish_generating
    done = [dec_state.num_open_nt == 0,
            len(dec_state.stack) == 2,
            gen_finish]
    return [all(done)]

