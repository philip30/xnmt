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
    super().__init__(src_reader=src_reader,
                     trg_reader=trg_reader,
                     encoder=encoder,
                     decoder=decoder)
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
    
  def add_input(self, prev_word: xnmt.Batch, state: models.UniDirectionalState) -> models.UniDirectionalState:
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
    losses = {}
    
    src = xnmt.mark_as_batch([s.get_unpadded_sent() for s in src])
    trg = xnmt.mark_as_batch([t.get_unpadded_sent() for t in trg])
    
    decoder_states_batch =[]
    for src_i, trg_i in zip(src, trg):
      decoder_states_batch.append(self.auto_regressive_states(src_i, trg_i))
     
    if self.train_nmt_mle:
      batch_losses = []
      for decoder_states, trg_i in zip(decoder_states_batch, trg):
        seq_states = nn.decoders.arb_len.ArbSeqLenUniDirectionalState(
          rnn_state=models.IdentityUniDirectionalState(
            dy.concatenate_to_batch([dec_state.output() for dec_state in decoder_states])
          ),
          context=dy.concatenate_to_batch([dec_state.context() for dec_state in decoder_states]),
          attender_state= None,
          timestep= -1,
          src=None
        )
        seq_ref = xnmt.mark_as_batch(trg_i.words)
        batch_losses.append(dy.sum_batches(self.decoder.calc_loss(seq_states, seq_ref)))
      dy.forward(batch_losses)
      losses["simult_nmt_mle"] = xnmt.LossExpr(
        expr=dy.concatenate_to_batch(batch_losses),
        units=[trg[i].len_unpadded() for i in range(trg.batch_size())]
      )
    if self.train_pol_mle and self.policy_agent is not None:
      batch_losses = []
      units = []
      for decoder_states, src_i in zip(decoder_states_batch, src):
        decoder_states = list(reversed(decoder_states[-1].collect_trajectories_backward()))
        seq_states = models.IdentityUniDirectionalState(
          dy.concatenate_to_batch([dec_state.network_state.output() for dec_state in decoder_states])
        )
        seq_ref = xnmt.mark_as_batch(src_i.oracle)
        seq_log_softmax = dy.concatenate_to_batch([dec_state.simult_action.log_softmax for dec_state in decoder_states])
        batch_losses.append(dy.sum_batches(self.policy_agent.calc_loss(seq_states, seq_ref, seq_log_softmax)))
        units.append(len(decoder_states))
      dy.forward(batch_losses)
      losses["simult_pol_mle"] = xnmt.LossExpr(
        expr=dy.concatenate_to_batch(batch_losses),
        units=units
      )
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
      word = None if t == 0 else trg[t]
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
      reset_attender=True,
      network_state=network_state,
      parent=state
    )
  
  def _perform_write(self, state, search_action, prev_word, network_state) -> agents.SimultSeqLenUniDirectionalState:
    if state.decoder_state is None:
      decoder_state = self.decoder.initial_state(state.encodings_to_now(), state.src)
    elif state.reset_attender:
      decoder_state = nn.decoders.arb_len.ArbSeqLenUniDirectionalState(
        rnn_state=state.decoder_state.rnn_state,
        context=state.decoder_state.context(),
        attender_state=self.decoder.attender.initial_state(state.encodings_to_now().encode_seq),
        timestep=state.decoder_state.timestep,
        src=state.decoder_state.src
      )
    else:
      decoder_state = state.decoder_state
    
    return agents.SimultSeqLenUniDirectionalState(
      src=state.src,
      full_encodings=state.full_encodings,
      decoder_state=self.decoder.add_input(decoder_state, prev_word),
      num_reads=state.num_reads,
      num_writes=state.num_writes+1,
      simult_action=search_action,
      reset_attender=False,
      network_state=network_state,
      parent=state
    )
