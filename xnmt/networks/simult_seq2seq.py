import dynet as dy
import numpy as np

import xnmt
import xnmt.models as models
import xnmt.modules.nn as nn
import xnmt.rl.agents as agents
import xnmt.networks.seq2seq as base

from typing import Any

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


  def initial_state(self, src: xnmt.Batch, force_oracle=False) -> agents.SimultSeqLenUniDirectionalState:
    oracle_batch = None
    trg_count = None
    if type(src[0]) == xnmt.structs.sentences.OracleSentence:
      oracle = [src[i].oracle for i in range(src.batch_size())]
      trg_count = [sum([1 for x in src[i].oracle if x == agents.SimultPolicyAgent.WRITE]) for i in range(src.batch_size())]
      oracle_batch = xnmt.structs.batchers.pad(oracle)

    encoding = self.encoder.encode(src)
    encoder_seqs = encoding.encode_seq
    return agents.SimultSeqLenUniDirectionalState(
      oracle_batch=oracle_batch, src=src, full_encodings=encoder_seqs, network_state=self.policy_agent.initial_state(src),
      trg_counts=trg_count
    )

  def add_input(self, prev_word: xnmt.Batch, state: models.UniDirectionalState) -> agents.SimultSeqLenUniDirectionalState:
    if prev_word is not None and not xnmt.is_batched(prev_word):
      prev_word = xnmt.mark_as_batch([prev_word])

    while not self.finish_generating(prev_word, state):
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
    state = self.initial_state(src, force_oracle=True)
    word_pad = xnmt.structs.vocabs.Vocab.PAD

    if hasattr(state, "oracle_batch"):
      for s, t, o in zip(src, trg, state.oracle_batch):
        s = s.get_unpadded_sent()
        t = t.get_unpadded_sent()
        o = o.get_unpadded_sent()
        assert o.sent_len() == s.sent_len() + t.sent_len(), \
          "Expecting {} + {} but got {} for oracle len.\nSRC: {}\nTRG: {}\nORC: {}".format(
            s.sent_len(), t.sent_len(), o.sent_len(), str(s), str(t), str(o))

    mle_loss = []
    pol_loss = []
    while not self.finish_generating(-1, state):
      search_action, network_state = self.policy_agent.next_action(state)
      action_set = set(search_action.action_id)
      #print(search_action.action_id)
      if agents.SimultPolicyAgent.READ in action_set:
        new_state = self._perform_read(state, search_action, network_state)
        num_reads = new_state.num_reads
      else:
        num_reads = state.num_reads

      if agents.SimultPolicyAgent.WRITE in action_set:
        write_flag = np.zeros((trg.batch_size(), 1), dtype=int)
        write_flag[search_action.action_id == agents.SimultPolicyAgent.WRITE] = 1

        prev_word = [trg[i][state.num_writes[i]-1] \
                       if write_flag[i] > 0 and state.num_writes[i] > 0 \
                       else word_pad for i in range(trg.batch_size())]
        prev_mask = np.array([[1 if word == word_pad else 0 for word in prev_word]])
        prev_word = xnmt.mark_as_batch(data=prev_word, mask=xnmt.Mask(prev_mask.transpose()))


        ref_word = [trg[i][state.num_writes[i]] \
                      if write_flag[i] else word_pad for i in range(trg.batch_size())]
        ref_mask = np.array([[1 if word == word_pad else 0 for word in ref_word]])
        ref_word = xnmt.mark_as_batch(data=ref_word, mask=xnmt.Mask(ref_mask.transpose()))

        new_state =  self._perform_write(state, search_action, prev_word, network_state, np.asarray(1-ref_mask[0], int))
        #if state.decoder_state is not None:
        #  print(state.decoder_state.attender_state.read_mask.np_arr)
        num_writes = new_state.num_writes
        decoder_state = new_state.decoder_state

        if self.train_nmt_mle:
          ref_word = xnmt.mark_as_batch(data=ref_word, mask=xnmt.Mask(1-write_flag))
          loss = self.decoder.calc_loss(decoder_state, ref_word)
          if ref_word.mask is not None:
            loss = ref_word.mask.cmult_by_timestep_expr(loss, 0, True)

          mle_loss.append(loss)
        # BEGIN(DEBUG)
#        print("{} {} {} {} {} {}".format(search_action.action_id, mle_loss[-1].npvalue(), prev_word,
#                                      prev_word.mask.np_arr.transpose(), ref_word,
#                                      ref_word.mask.np_arr.transpose()))
        # END(DEBUG)
      else:
        num_writes = state.num_writes
        decoder_state = state.decoder_state
        write_flag = np.zeros((trg.batch_size(), 1), dtype=int)
        write_flag[search_action.action_id == agents.SimultPolicyAgent.WRITE] = 1

      if self.train_pol_mle:
        action_pad = xnmt.structs.vocabs.SimultActionVocab.PAD
        search_action = search_action.action_id
        oracle_action = [oracle[state.timestep] for oracle in state.oracle_batch]
        mask = xnmt.Mask(np.array([[1 if ref == action_pad else 0 for ref in oracle_action]]).transpose())
        oracle_batch = xnmt.mark_as_batch(oracle_action, mask)
        loss = self.policy_agent.calc_loss(network_state, oracle_batch)

        if oracle_batch.mask is not None:
          loss = oracle_batch.mask.cmult_by_timestep_expr(loss, 0, True)
        pol_loss.append(loss)

      state = agents.SimultSeqLenUniDirectionalState(
        src=state.src,
        full_encodings=state.full_encodings,
        oracle_batch=state.oracle_batch,
        decoder_state=decoder_state,
        num_reads=num_reads,
        num_writes=num_writes,
        simult_action=search_action,
        read_was_performed=agents.SimultPolicyAgent.READ in action_set,
        network_state=network_state,
        parent=state,
        force_oracle=state.force_oracle,
        timestep=state.timestep+1,
        trg_counts=state.trg_counts
      )

    total_loss = {}
    if mle_loss:
      total_loss["p(e|f)"] = xnmt.LossExpr(dy.esum(mle_loss), units=state.num_writes)
    if pol_loss:
      units = [src[i].oracle.len_unpadded() for i in range(src.batch_size())]
      total_loss["p(a|h)"] = xnmt.LossExpr(dy.esum(pol_loss), units=units)
    return xnmt.FactoredLossExpr(total_loss)

  def finish_generating(self, output: Any, dec_state: agents.SimultSeqLenUniDirectionalState):
    if dec_state.force_oracle or \
        (xnmt.is_train() and self.policy_agent.oracle_in_train) or \
        (not xnmt.is_train() and  self.policy_agent.oracle_in_test):
      return self.policy_agent.finish_generating(dec_state)
    return super().finish_generating(output, dec_state)

  def _perform_read(self,
                    state: agents.SimultSeqLenUniDirectionalState,
                    search_action: models.SearchAction,
                    network_state: models.PolicyAgentState) -> agents.SimultSeqLenUniDirectionalState:
    read_inc = np.zeros(search_action.action_id.shape, dtype=int)
    read_inc[search_action.action_id == agents.SimultPolicyAgent.READ] = 1

    return agents.SimultSeqLenUniDirectionalState(
      src=state.src,
      full_encodings=state.full_encodings,
      decoder_state=state.decoder_state,
      num_reads=state.num_reads+read_inc,
      num_writes=state.num_writes,
      simult_action=search_action,
      read_was_performed=True,
      network_state=network_state,
      oracle_batch=state.oracle_batch,
      timestep=state.timestep+1,
      force_oracle=state.force_oracle,
      parent=state,
      trg_counts=state.trg_counts
    )

  def _perform_write(self,
                     state: agents.SimultSeqLenUniDirectionalState,
                     search_action: models.SearchAction,
                     prev_word: xnmt.Batch,
                     network_state: models.PolicyAgentState,
                     write_flag: np.ndarray = np.asarray([1], int)) -> agents.SimultSeqLenUniDirectionalState:
    if state.decoder_state is None:
      decoder_state = self.decoder.initial_state(models.EncoderState(state.full_encodings, None), state.src)
    else:
      decoder_state = state.decoder_state

    if (state.read_was_performed or state.decoder_state is None) and \
        hasattr(self.decoder, "attender") and self.decoder.attender is not None:
      attender_state = decoder_state.attender_state
      read_masks = np.ones((state.src.batch_size(), state.src.sent_len()), dtype=float)
      for num_read, read_mask in zip(state.num_reads, read_masks):
        read_mask[:num_read] = 0

      decoder_state = nn.decoders.arb_len.ArbSeqLenUniDirectionalState(
        rnn_state=decoder_state.rnn_state,
        context=decoder_state.context(),
        attender_state=models.AttenderState(
          curr_sent=attender_state.curr_sent,
          sent_context=attender_state.sent_context,
          input_mask=attender_state.input_mask,
          read_mask=xnmt.Mask(read_masks),
          attention=attender_state.attention
        ),
        timestep=decoder_state.timestep,
        src=decoder_state.src
      )
    num_writes = state.num_writes + write_flag
    equal_to_1 = np.logical_and(num_writes == 1, write_flag == 1)
    if any(equal_to_1):
      first_mask = np.ones_like(num_writes, dtype=int)
      first_mask[equal_to_1] = 0
      first_mask = xnmt.Mask(np.expand_dims(first_mask.transpose(), axis=1))
    else:
      first_mask = None

    return agents.SimultSeqLenUniDirectionalState(
      src=state.src,
      full_encodings=state.full_encodings,
      decoder_state=self.decoder.add_input(decoder_state, prev_word, first_mask),
      num_reads=state.num_reads,
      num_writes=num_writes,
      simult_action=search_action,
      read_was_performed=False,
      network_state=network_state,
      oracle_batch=state.oracle_batch,
      force_oracle=state.force_oracle,
      timestep=state.timestep+1,
      parent=state,
      trg_counts=state.trg_counts
    )
