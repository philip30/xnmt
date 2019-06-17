import dynet as dy
import numpy as np

import xnmt
import functools
import xnmt.models as models
import xnmt.modules.nn as nn
import xnmt.rl.agents as agents
import xnmt.networks.seq2seq as base
from xnmt.cython import xnmt_cython


from typing import Any, Optional


class SimultSeq2Seq(base.Seq2Seq, xnmt.Serializable):
  yaml_tag = "!SimultSeq2Seq"

  @xnmt.serializable_init
  def __init__(self,
               src_reader: models.InputReader,
               trg_reader: models.InputReader,
               encoder: models.Encoder = xnmt.bare(nn.SeqEncoder),
               decoder: models.Decoder = xnmt.bare(nn.ArbLenDecoder),
               policy_agent: agents.SimultPolicyAgent = xnmt.bare(agents.SimultPolicyAgent),
               train_nmt_mle: Optional[bool] = True,
               train_pol_mle: Optional[bool] = True,
               baseline_network: Optional[models.Transform] = None,
               default_layer_dim: int = xnmt.default_layer_dim):
    super().__init__(src_reader=src_reader, trg_reader=trg_reader, encoder=encoder, decoder=decoder)
    self.policy_agent = policy_agent
    self.train_nmt_mle = train_nmt_mle
    self.train_pol_mle = train_pol_mle
    self.baseline_network = self.add_serializable_component(
      "baseline_network", baseline_network, lambda: nn.Linear(default_layer_dim, 1)
    )

    if isinstance(decoder, nn.ArbLenDecoder):
      if not isinstance(decoder.bridge, nn.ZeroBridge):
        if isinstance(decoder.bridge, nn.NoBridge):
          decoder.bridge = nn.ZeroBridge(decoder.bridge.dec_layers, decoder.bridge.dec_dim)
        else:
          xnmt.logger.warning("Cannot use any bridge except ZeroBridge for SimultSeq2Seq.")


  def initial_state(self, src: xnmt.Batch, force_oracle=False) -> agents.SimultSeqLenUniDirectionalState:
    oracle_batch = None
    trg_count = None
    if type(src[0]) == xnmt.structs.sentences.OracleSentence:
      oracle = [src[i].oracle for i in range(src.batch_size())]
      trg_count = [sum([1 for x in src[i].oracle if x == agents.SimultPolicyAgent.WRITE \
                        or x == agents.SimultPolicyAgent.PREDICT_WRITE]) for i in range(src.batch_size())]
      oracle_batch = xnmt.structs.batchers.pad(oracle)

    encoding = self.encoder.encode(src)
    encoder_seqs = encoding.encode_seq
    decoder_init = self.decoder.initial_state(encoding, src)

    if isinstance(decoder_init, nn.decoders.arb_len.ArbSeqLenUniDirectionalState):
      decoder_init.attender_state.read_mask = xnmt.Mask(np.expand_dims(np.array([1] * src.batch_size()), axis=1))

    return agents.SimultSeqLenUniDirectionalState(
      oracle_batch=oracle_batch, src=src, full_encodings=encoder_seqs, network_state=self.policy_agent.initial_state(src),
      trg_counts=trg_count, decoder_state=decoder_init
    )

  def add_input(self, prev_word: xnmt.Batch, state: models.UniDirectionalState) -> agents.SimultSeqLenUniDirectionalState:
    if prev_word is not None and not xnmt.is_batched(prev_word):
      prev_word = xnmt.mark_as_batch([prev_word])

    while not self.finish_generating(prev_word, state):
      search_action, network_state = self.policy_agent.next_action(state)
      if search_action.action_id == agents.SimultPolicyAgent.READ or \
         search_action.action_id == agents.SimultPolicyAgent.PREDICT_READ:
        state = self._perform_read(state, search_action)
        state.network_state = network_state
      elif search_action.action_id == agents.SimultPolicyAgent.WRITE or \
           search_action.action_id == agents.SimultPolicyAgent.PREDICT_WRITE:
        state = self._perform_write(state, search_action, prev_word, np.asarray([1]))
        state.network_state = network_state
        break
      else:
        raise NotImplementedError()

    return state

  def _is_reading(self, a):
    if isinstance(a, set):
      return (agents.SimultPolicyAgent.READ in a) or (agents.SimultPolicyAgent.PREDICT_READ in a)
    elif isinstance(a, models.SearchAction):
      return a.action_id == agents.SimultPolicyAgent.READ or a.action_id == agents.SimultPolicyAgent.PREDICT_READ
    else:
      return NotImplementedError()

  def _is_writing(self, a):
    if isinstance(a, set):
      return (agents.SimultPolicyAgent.WRITE in a) or (agents.SimultPolicyAgent.PREDICT_WRITE in a)
    elif isinstance(a, models.SearchAction):
      return a.action_id == agents.SimultPolicyAgent.WRITE or a.action_id == agents.SimultPolicyAgent.PREDICT_WRITE
    else:
      return NotImplementedError()

  def calc_nll(self, src: xnmt.Batch, trg: xnmt.Batch):
    batch_size = src.batch_size()
    start_sym = xnmt.Vocab.SS
    state = self.initial_state(src)

    mle_loss = []
    pol_loss = []
    while not self.finish_generating(-1, state):
      search_action, network_state = self.policy_agent.next_action(state, force_oracle=True, is_sample=False)
      action_set = set(search_action.action_id)
      nwrs = state.num_writes

      ### READING ###
      if self._is_reading(action_set):
        read_state = self._perform_read(state, search_action)
      else:
        read_state = state

      ### WRITING ###
      is_writing = self._is_writing(action_set)
      write_mask = None
      if is_writing:
        write_flag = np.zeros(batch_size, dtype=int)
        is_write = search_action.action_id == agents.SimultPolicyAgent.WRITE
        is_pwrite = search_action.action_id == agents.SimultPolicyAgent.PREDICT_WRITE
        write_flag[np.logical_or(is_write, is_pwrite)] = 1
        write_mask = xnmt.Mask(1-np.expand_dims(write_flag, 1))

        prev_word = np.asarray([trg[i][nwrs[i]-1] if nwrs[i] > 0 else start_sym for i in range(batch_size)])
        prev_word = xnmt.mark_as_batch(prev_word, write_mask)
        write_state = self._perform_write(state, search_action, prev_word, write_flag)
      else:
        write_state = state

      ### Calculate MLE Loss ###
      if is_writing and self.train_nmt_mle:
        ref_word = [(trg[i][nwrs[i]] if nwrs[i] < trg.sent_len() else xnmt.Vocab.PAD) for i in range(batch_size)]
        ref_word = xnmt.mark_as_batch(data=ref_word, mask=write_mask)

        loss = self.decoder.calc_loss(write_state.decoder_state, ref_word)
        loss = write_mask.cmult_by_timestep_expr(loss, 0, True)
        mle_loss.append(loss)

      ### Calculate MLE POL Loss ###
      if self.train_pol_mle:
        oracle_action = np.asarray([oracle[state.timestep] for oracle in state.oracle_batch])
        oracle_mask = np.zeros(batch_size)
        oracle_mask[oracle_action == xnmt.structs.vocabs.SimultActionVocab.PAD] = 1
        oracle_mask = xnmt.Mask(np.expand_dims(oracle_mask, axis=1))
        oracle_batch = xnmt.mark_as_batch(oracle_action, oracle_mask)

        loss = self.policy_agent.calc_loss(network_state, oracle_batch)
        loss = oracle_mask.cmult_by_timestep_expr(loss, 0, True)
        pol_loss.append(loss)

      ### Next State ###
      state = agents.SimultSeqLenUniDirectionalState(
        src=state.src,
        full_encodings=state.full_encodings,
        oracle_batch=state.oracle_batch,
        decoder_state=write_state.decoder_state,
        num_reads=read_state.num_reads,
        num_writes=write_state.num_writes,
        simult_action=search_action,
        read_was_performed=read_state.read_was_performed,
        network_state=network_state,
        parent=state,
        timestep=state.timestep+1,
        trg_counts=state.trg_counts
      )
    # END: Loop
    ### Calculate Total Loss
    total_loss = {}
    if mle_loss:
      total_loss["p(e|f)"] = xnmt.LossExpr(dy.esum(mle_loss), units=state.num_writes)
    if pol_loss:
      units = [src[i].oracle.len_unpadded() for i in range(src.batch_size())]
      total_loss["p(a|h)"] = xnmt.LossExpr(dy.esum(pol_loss), units=units)
    return xnmt.FactoredLossExpr(total_loss)


  def calc_reinforce_loss(self,
                          src: xnmt.Batch,
                          trg: xnmt.Batch,
                          num_sample=1,
                          max_len=100):
    self.policy_agent.oracle_in_train = False
    batch_size = src.batch_size()
    refs = [trg[i].get_unpadded_sent().words for i in range(trg.batch_size())]

    reinf_losses = []
    basel_losses = []
    # BEGIN LOOP: Sample
    for _ in range(num_sample):
      state = self.initial_state(src)
      done = np.array([False] * src.batch_size())
      nwrs = state.num_writes

      actions = []
      words = [[xnmt.Vocab.SS] for _ in range(batch_size)]
      log_ll = []
      baseline_inp = []
      baseline_flg = []

      # BEGIN LOOP: Create trajectory
      while not np.all(done):
        search_action, network_state = self.policy_agent.next_action(state, is_sample=True)
        done_mask = dy.inputTensor([0 if done[i] else 1 for i in range(batch_size)], batched=True)
        search_action.action_id[done] = xnmt.structs.vocabs.SimultActionVocab.PAD
        log_ll.append(dy.cmult(search_action.log_likelihood, done_mask))
        actions.append(search_action.action_id)

        ### Baseline
        bs_inp = state.network_state.output() or dy.zeros(*network_state.output().dim())
        baseline_inp.append(self.baseline_network.transform(dy.nobackprop(bs_inp)))
        baseline_flg.append(done_mask)

        ### PERFORM READING + WRITING
        action_set = set(search_action.action_id)
        ### READING ###
        if self._is_reading(action_set):
          read_state = self._perform_read(state, search_action)
        else:
          read_state = state

        ### WRITING ###
        is_writing = self._is_writing(action_set)
        if is_writing:
          write_flag = np.zeros(batch_size, dtype=int)
          write_flag[search_action.action_id == agents.SimultPolicyAgent.WRITE] = 1
          write_flag[search_action.action_id == agents.SimultPolicyAgent.PREDICT_WRITE] = 1
          write_mask = xnmt.Mask(1-np.expand_dims(write_flag, 1))

          prev_word = np.asarray([words[i][nwrs[i]] for i in range(batch_size)])
          prev_word = xnmt.mark_as_batch(prev_word, write_mask)
          write_state = self._perform_write(state, search_action, prev_word, write_flag)

          search_action = self.best_k(write_state, 1, True)
          for i in range(batch_size):
            if write_flag[i]:
              word = search_action[0].action_id[i]
              words[i].append(word)
              done[i] = len(words[i]) - 1 >= max_len or word == self.decoder.eog_symbol

        else:
          write_state = state

        ### Next State ###
        state = agents.SimultSeqLenUniDirectionalState(
          src=state.src,
          full_encodings=state.full_encodings,
          oracle_batch=state.oracle_batch,
          decoder_state=write_state.decoder_state,
          num_reads=read_state.num_reads,
          num_writes=write_state.num_writes,
          simult_action=search_action,
          read_was_performed=read_state.read_was_performed,
          network_state=network_state,
          parent=state,
          timestep=state.timestep+1,
          trg_counts=state.trg_counts
        )
      # END LOOP: Create trajectory
      ### Calculate Rewards ###
      words = [w[1:] for w in words]
      bleus = [10 * np.asarray(xnmt_cython.bleu_sentence_prog(4, 1, ref_i, hyp_i)) for ref_i, hyp_i in zip(refs, words)]
      tr_bleus = []

      actions = np.asarray(actions).transpose()
      reward = np.zeros_like(actions, dtype=float)
      for i, bleu in enumerate(bleus):
        true_bleu = bleu[-1]
        now_bleu = bleu
        shf_bleu = np.roll(bleu, shift=True)
        shf_bleu[0] = 0
        diff = now_bleu - shf_bleu
        diff[-1] = true_bleu
        k = 0
        for j in range(len(actions[i])):
          if actions[i][j] == 5 or actions[i][j] == 7:
            len_reward = 1 if k < trg[i].len_unpadded() else 0
            reward[i][j] = diff[k] + len_reward
            k += 1
        reward[i] = reward[i][::-1].cumsum()[::-1]
        tr_bleus.append(true_bleu)
      reward = dy.inputTensor(np.asarray(reward).transpose(), batched=True)

      ### Reward Discount ###
      baseline = dy.concatenate(baseline_inp, d=0)
      reward = reward - baseline

      ### Variance Reduction ###
      z_normalization = False
      if z_normalization:
        r_mean = dy.mean_dim(reward, d=[0], b=False)
        r_std = dy.std_dim(reward, d=[0], b=False)
        reward = dy.cdiv((reward - r_mean), r_std + xnmt.globals.EPS)

      ### calculate loss ###
      reward = dy.nobackprop(reward)
      log_ll = dy.concatenate(log_ll, d=0)
      rf_loss = -1 * dy.cmult(reward, log_ll)
      rf_units = [len(x) for (x) in words]
      reinf_losses.append(xnmt.LossExpr(dy.sum_elems(rf_loss), rf_units))

      flags = dy.concatenate(baseline_flg, d=0)
      baseline_loss = dy.squared_distance(baseline, reward)
      baseline_loss = dy.cmult(baseline_loss, flags)
      baseline_units = np.sum(flags.npvalue(), axis=0)
      basel_losses.append(xnmt.LossExpr(dy.sum_elems(baseline_loss), baseline_units))

      print("BLEU: {}, LOG LL: {}".format(np.mean(tr_bleus),
                                          np.mean(dy.cdiv(dy.sum_elems(log_ll), dy.inputTensor(baseline_units, batched=True)).value())))
    # END LOOP: Sample
    rf_loss = functools.reduce(lambda x, y: x+y, reinf_losses)
    bs_loss = functools.reduce(lambda x, y: x+y,basel_losses)
    return xnmt.FactoredLossExpr({"rf_loss": rf_loss, "bs_loss": bs_loss})

  def finish_generating(self, output: Any, dec_state: agents.SimultSeqLenUniDirectionalState):
    if (self.policy_agent.oracle_in_train and xnmt.is_train()) or \
       (self.policy_agent.oracle_in_test and not xnmt.is_train()):
      return self.policy_agent.finish_generating(dec_state)
    return super().finish_generating(output, dec_state)

  def _perform_read(self,
                    state: agents.SimultSeqLenUniDirectionalState,
                    search_action: models.SearchAction) -> agents.SimultSeqLenUniDirectionalState:
    read_flag = np.zeros(state.num_reads.shape, dtype=int)
    is_read = search_action.action_id == agents.SimultPolicyAgent.READ
    is_pread = search_action.action_id == agents.SimultPolicyAgent.PREDICT_READ
    read_flag[np.logical_or(is_read, is_pread)] = 1

    return agents.SimultSeqLenUniDirectionalState(
      src=state.src,
      full_encodings=state.full_encodings,
      decoder_state=state.decoder_state,
      num_reads=state.num_reads+read_flag,
      num_writes=state.num_writes,
      simult_action=search_action,
      read_was_performed=True,
      network_state=state.network_state,
      oracle_batch=state.oracle_batch,
      timestep=state.timestep+1,
      parent=state,
      trg_counts=state.trg_counts
    )

  def _perform_write(self,
                     state: agents.SimultSeqLenUniDirectionalState,
                     search_action: models.SearchAction,
                     prev_word: xnmt.Batch,
                     write_flag: np.ndarray) -> agents.SimultSeqLenUniDirectionalState:
    decoder_state = state.decoder_state

    if state.read_was_performed and hasattr(self.decoder, "attender") and self.decoder.attender is not None:
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
        src=decoder_state.src,
        prev_embedding=decoder_state.prev_embedding,
        position=decoder_state.position,
        merged_context=decoder_state.merged_context
      )
    return agents.SimultSeqLenUniDirectionalState(
      src=state.src,
      full_encodings=state.full_encodings,
      decoder_state=self.decoder.add_input(decoder_state, prev_word),
      num_reads=state.num_reads,
      num_writes=state.num_writes + write_flag,
      simult_action=search_action,
      read_was_performed=False,
      network_state=state.network_state,
      oracle_batch=state.oracle_batch,
      timestep=state.timestep+1,
      parent=state,
      trg_counts=state.trg_counts
    )
