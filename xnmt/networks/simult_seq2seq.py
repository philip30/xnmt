import dynet as dy

import xnmt
import xnmt.models as models
import xnmt.modules.nn as nn
import xnmt.rl.agents as agents
import xnmt.networks.seq2seq as base

from typing import List


class SimultSeq2Seq(base.Seq2Seq, xnmt.Serializable):
  yaml_tag = "!SimultSeq2Seq"

  @xnmt.serializable_init
  def __init__(self,
               src_reader: models.InputReader,
               trg_reader: models.InputReader,
               encoder: models.Encoder = xnmt.bare(nn.SeqEncoder),
               decoder: models.Decoder = xnmt.bare(nn.ArbLenDecoder),
               policy_agent: agents.SimultPolicyAgent = xnmt.bare(agents.SimultPolicyAgent),
               train_nmt_mle: bool = True,
               train_pol_mle: bool = True):
    super().__init__(src_reader=src_reader, trg_reader=trg_reader, encoder=encoder, decoder=decoder)
    self.policy_agent = policy_agent
    self.train_nmt_mle = train_nmt_mle
    self.train_pol_mle = train_pol_mle
    
    if isinstance(decoder, nn.ArbLenDecoder):
      if not isinstance(decoder.bridge, nn.NoBridge):
        xnmt.logger.warning("Cannot use any bridge except no bridge for SimultSeq2Seq.")
  

  def initial_state(self, src: xnmt.Batch) -> agents.SimultSeqLenUniDirectionalState:
    assert src.batch_size() == 1
    encoding = self.encoder.encode(src)
    encoder_seqs = encoding.encode_seq
    return agents.SimultSeqLenUniDirectionalState(
      src=src, full_encodings=encoder_seqs, network_state=self.policy_agent.initial_state(src)
    )
    
  def add_input(self, prev_word: xnmt.Batch, state: models.UniDirectionalState) -> agents.SimultSeqLenUniDirectionalState:
    while True:
      search_action, network_state = self.policy_agent.next_action(state)
      
      if search_action.action_id == agents.SimultPolicyAgent.READ:
        state = self._perform_read(state, search_action, network_state)
      elif search_action.action_id == agents.SimultPolicyAgent.WRITE:
        state = self._perform_write(state, search_action, prev_word, network_state)
        break
      else:
        raise ValueError()
      
    return state

  def calc_nll(self, src: xnmt.Batch, trg: xnmt.Batch):
    # Trajectories
    decoder_states = [self.auto_regressive_states(src[i].get_unpadded_sent(), trg[i].get_unpadded_sent()) \
                      for i in range(src.batch_size())]
    # Calc Loss
    losses = {}
    if self.train_nmt_mle:
      batch_losses = [dy.esum([self.decoder.calc_loss(decoder_states[i][j], trg[i][j]) \
                               for j in range(len(decoder_states[i]))]
                              ) \
                      for i in range(trg.batch_size())]
      units = [trg[i].len_unpadded() for i in range(trg.batch_size())]
      losses["simult_nmt"] = xnmt.LossExpr(expr=dy.concatenate_to_batch(batch_losses), units=units)

    if self.train_pol_mle and self.policy_agent.policy_network is not None:
      batch_losses = []
      units = []
      for dec_state, src_sent in zip(decoder_states, src):
        loss = []
        for i, decoder_state in enumerate(dec_state[-1].collect_trajectories_backward()):
          loss.append(self.policy_agent.calc_loss(decoder_state.network_state,
                                                  src_sent.oracle[-i-1],
                                                  decoder_state.simult_action.log_softmax))
        batch_losses.append(dy.esum(loss))
        units.append(len(loss))
      losses["simult_pol"] = xnmt.LossExpr(expr=dy.concatenate_to_batch(batch_losses), units=units)
    
    return xnmt.FactoredLossExpr(losses)
  
  def finish_generating(self, output: int, dec_state: agents.SimultSeqLenUniDirectionalState):
    if dec_state.force_oracle or \
        xnmt.is_train() and self.policy_agent.oracle_in_train or \
        (not xnmt.is_train()) and  self.policy_agent.oracle_in_test:
      return self.policy_agent.finish_generating(dec_state)
    return super().finish_generating(output, dec_state)
  
  def auto_regressive_states(self, src: xnmt.Sentence, trg: xnmt.Sentence) \
      -> List[agents.SimultSeqLenUniDirectionalState]:
    assert src.batch_size() == 1 and trg.batch_size() == 1
    src = xnmt.mark_as_batch([src])
    state = self.initial_state(src)
    states = []
    for t in range(trg.sent_len()):
      word = None if t == 0 else trg[t-1]
      state = self.add_input(word, state)
      states.append(state)
    return states
 
  def _perform_read(self, state, search_action, network_state) -> agents.SimultSeqLenUniDirectionalState:
    return agents.SimultSeqLenUniDirectionalState(
      src=state.src,
      full_encodings=state.full_encodings,
      decoder_state=state.decoder_state,
      num_reads=state.num_reads+1,
      num_writes=state.num_writes,
      simult_action=search_action,
      read_was_performed=True,
      network_state=network_state,
      parent=state
    )
  
  def _perform_write(self, state: agents.SimultSeqLenUniDirectionalState, search_action, prev_word, network_state) -> agents.SimultSeqLenUniDirectionalState:
    if state.decoder_state is None:
      decoder_state = self.decoder.initial_state(models.EncoderState(state.full_encodings, None), state.src)
    else:
      decoder_state = state.decoder_state
   
    if state.read_was_performed or state.decoder_state is None:
      attender_state = decoder_state.attender_state
      decoder_state = nn.decoders.arb_len.ArbSeqLenUniDirectionalState(
        rnn_state=decoder_state.rnn_state,
        context=decoder_state.context(),
        attender_state=models.AttenderState(
          curr_sent=dy.pick_range(attender_state.initial_context[0], 0, state.num_reads, d=1),
          sent_context=dy.pick_range(attender_state.initial_context[1], 0, state.num_reads, d=1),
          input_mask=attender_state.input_mask,
          attention=attender_state.attention,
          initial_context=attender_state.initial_context
        ),
        timestep=decoder_state.timestep,
        src=decoder_state.src
      )
    
    return agents.SimultSeqLenUniDirectionalState(
      src=state.src,
      full_encodings=state.full_encodings,
      decoder_state=self.decoder.add_input(decoder_state, prev_word),
      num_reads=state.num_reads,
      num_writes=state.num_writes+1,
      simult_action=search_action,
      read_was_performed=False,
      network_state=network_state,
      parent=state
    )
