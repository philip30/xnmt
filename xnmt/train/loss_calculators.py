from typing import Optional, Sequence

import xnmt
import xnmt.models as models


class MLELoss(models.LossCalculator, xnmt.Serializable):
  yaml_tag = "!MLELoss"
  """
  Max likelihood loss calculator.
  """
  @xnmt.serializable_init
  def __init__(self) -> None:
    pass

  def _perform_calc_loss(self,
                         model: models.ConditionedModel,
                         src: xnmt.Batch,
                         trg: xnmt.Batch) -> xnmt.FactoredLossExpr:
    return xnmt.FactoredLossExpr({"mle": model.calc_nll(src, trg)})

class ReinforceLoss(models.LossCalculator, xnmt.Serializable):
  yaml_tag = "!ReinforceLoss"

  @xnmt.serializable_init
  def __init__(self, num_sample: int = 1, max_len: int = 100):
    self.num_sample = num_sample
    self.max_len = max_len

  def _perform_calc_loss(self,
                         model: models.ConditionedModel,
                         src: xnmt.Batch,
                         trg: xnmt.Batch):
    return model.calc_reinforce_loss(src, trg, self.num_sample, self.max_len)


class CompositeLoss(models.LossCalculator, xnmt.Serializable):
  yaml_tag = "!CompositeLoss"
  """
  Summing losses from multiple LossCalculator.
  """
  @xnmt.serializable_init
  def __init__(self, losses: Sequence[models.LossCalculator], loss_weight: Optional[Sequence[float]] = None) -> None:
    self.losses = losses
    if loss_weight is None:
      self.loss_weight = [1.0 for _ in range(len(losses))]
    else:
      self.loss_weight = loss_weight
    assert len(self.loss_weight) == len(losses)


  def _perform_calc_loss(self,
                         model: models.ConditionedModel,
                         src: xnmt.Batch,
                         trg: xnmt.Batch) -> xnmt.FactoredLossExpr:
    total_loss = {}
    for i, (loss, weight) in enumerate(zip(self.losses, self.loss_weight)):
      total_loss[str(i)] = loss._perform_calc_loss(model, src, trg) * weight
    return xnmt.FactoredLossExpr(total_loss)


#class ReinforceLoss(Serializable, LossCalculator):
#  """
#  Reinforce Loss according to Ranzato+, 2015.
#  SEQUENCE LEVEL TRAINING WITH RECURRENT NEURAL NETWORKS.
#
#  (This is not the MIXER algorithm)
#
#  https://arxiv.org/pdf/1511.06732.pdf
#  """
#  @xnmt.serializable_init
#  def __init__(self,
#               baseline: Optional[Serializable]=None,
#               evaluation_metric: metrics.SentenceLevelEvaluator = bare(metrics.FastBLEUEvaluator),
#               search_strategy: search_strategies.SearchStrategy = bare(search_strategies.SamplingSearch),
#               inv_eval: bool = True,
#               decoder_hidden_dim: numbers.Integral = Ref("exp_global.default_layer_dim")) -> None:
#    self.inv_eval = inv_eval
#    self.search_strategy = search_strategy
#    self.evaluation_metric = evaluation_metric
#    self.baseline = self.add_serializable_component("baseline", baseline,
#                                                    lambda: transforms.Linear(input_dim=decoder_hidden_dim, output_dim=1))
#
#  def _perform_calc_loss(self,
#                         model: 'model_base.ConditionedModel',
#                         src: Union[sentences.Sentence, 'batchers.Batch'],
#                         trg: Union[sentences.Sentence, 'batchers.Batch']) -> losses.FactoredLossExpr:
#    search_outputs = model.generate_search_output(src, self.search_strategy)
#    sign = -1 if self.inv_eval else 1
#
#    # TODO: Fix units
#    total_loss = collections.defaultdict(int)
#    for search_output in search_outputs:
#      # Calculate rewards
#      eval_score = []
#      for trg_i, sample_i in zip(trg, search_output.word_ids):
#        # Removing EOS
#        sample_i = utils.remove_eos(sample_i.tolist(), vocabs.Vocab.ES)
#        ref_i = trg_i.words[:trg_i.len_unpadded()]
#        score = self.evaluation_metric.evaluate_one_sent(ref_i, sample_i)
#        eval_score.append(sign * score)
#      reward = dy.inputTensor(eval_score, batched=True)
#      # Composing losses
#      baseline_loss = []
#      cur_losses = []
#      for state, mask in zip(search_output.state, search_output.mask):
#        bs_score = self.baseline.transform(dy.nobackprop(state.as_vector()))
#        baseline_loss.append(dy.squared_distance(reward, bs_score))
#        logsoft = model.decoder.scorer.calc_log_probs(state.as_vector())
#        loss_i = dy.cmult(logsoft, reward - bs_score)
#        cur_losses.append(dy.cmult(loss_i, dy.inputTensor(mask, batched=True)))
#
#      total_loss["polc_loss"] += dy.sum_elems(dy.esum(cur_losses))
#      total_loss["base_loss"] += dy.sum_elems(dy.esum(baseline_loss))
#    units = [t.len_unpadded() for t in trg]
#    total_loss = losses.FactoredLossExpr({k: losses.LossExpr(v, units) for k, v in total_loss.items()})
#    return losses.FactoredLossExpr({"risk": total_loss})
#
#
#class MinRiskLoss(Serializable, LossCalculator):
#
#  @xnmt.serializable_init
#  def __init__(self,
#               evaluation_metric: metrics.Evaluator = bare(metrics.FastBLEUEvaluator),
#               alpha: numbers.Real = 0.005,
#               inv_eval: bool = True,
#               unique_sample: bool = True,
#               search_strategy: search_strategies.SearchStrategy = bare(search_strategies.SamplingSearch)) -> None:
#    # Samples
#    self.alpha = alpha
#    self.evaluation_metric = evaluation_metric
#    self.inv_eval = inv_eval
#    self.unique_sample = unique_sample
#    self.search_strategy = search_strategy
#
#  def _perform_calc_loss(self,
#                         model: 'model_base.ConditionedModel',
#                         src: Union[sentences.Sentence, 'batchers.Batch'],
#                         trg: Union[sentences.Sentence, 'batchers.Batch']) -> losses.FactoredLossExpr:
#    batch_size = trg.batch_size()
#    uniques = [set() for _ in range(batch_size)]
#    deltas = []
#    probs = []
#    sign = -1 if self.inv_eval else 1
#    search_outputs = model.generate_search_output(src, self.search_strategy)
#    # TODO: Fix this
#    for search_output in search_outputs:
#      assert len(search_output.word_ids) == 1
#      assert search_output.word_ids[0].shape == (len(search_output.state),)
#      logprob = []
#      for word, state in zip(search_output.word_ids[0], search_output.state):
#        lpdist = model.decoder.scorer.calc_log_probs(state.as_vector())
#        lp = dy.pick(lpdist, word)
#        logprob.append(lp)
#      sample = search_output.word_ids
#      logprob = dy.esum(logprob) * self.alpha
#      # Calculate the evaluation score
#      eval_score = np.zeros(batch_size, dtype=float)
#      mask = np.zeros(batch_size, dtype=float)
#      for j in range(batch_size):
#        ref_j = utils.remove_eos(trg[j].words, vocabs.Vocab.ES)
#        hyp_j = utils.remove_eos(sample[j].tolist(), vocabs.Vocab.ES)
#        if self.unique_sample:
#          hash_val = hash(tuple(hyp_j))
#          if len(hyp_j) == 0 or hash_val in uniques[j]:
#            mask[j] = -1e20 # represents negative infinity
#            continue
#          else:
#            uniques[j].add(hash_val)
#          # Calc evaluation score
#        eval_score[j] = self.evaluation_metric.evaluate_one_sent(ref_j, hyp_j) * sign
#      # Appending the delta and logprob of this sample
#      prob = logprob + dy.inputTensor(mask, batched=True)
#      deltas.append(dy.inputTensor(eval_score, batched=True))
#      probs.append(prob)
#    sample_prob = dy.softmax(dy.concatenate(probs))
#    deltas = dy.concatenate(deltas)
#    risk = dy.sum_elems(dy.cmult(sample_prob, deltas))
#    units = [t.len_unpadded() for t in trg]
#    return losses.FactoredLossExpr({"risk": losses.LossExpr(risk, units)})
#class GlobalFertilityLoss(models.LossCalculator, xnmt.Serializable):
#  yaml_tag = "!GlobalFertilityLoss"
#  """
#  A fertility loss according to Cohn+, 2016.
#  Incorporating Structural Alignment Biases into an Attentional Neural Translation Model
#
#  https://arxiv.org/pdf/1601.01085.pdf
#  """
#  @xnmt.serializable_init
#  def __init__(self) -> None:
#    pass
#
#  def _perform_calc_loss(self,
#                         model: models.ConditionedModel,
#                         src: xnmt.Batch,
#                         trg: xnmt.Batch) -> xnmt.FactoredLossExpr:
#    assert hasattr(model, "attender") and hasattr(model.attender, "attention_vecs"), \
#           "Must be called after MLELoss with networks that have attender."
#
#    masked_attn = model.attender.attention_vecs
#    if trg.mask is not None:
#      trg_mask = 1-(trg.mask.np_arr.transpose())
#      masked_attn = [dy.cmult(attn, dy.inputTensor(mask, batched=True)) for attn, mask in zip(masked_attn, trg_mask)]
#    loss = dy.sum_elems(dy.square(1 - dy.esum(masked_attn)))
#    units = [t.len_unpadded() for t in trg]
#    return xnmt.FactoredLossExpr({"global_fertility": xnmt.LossExpr(loss, units)})
#
#
#
#
#
#
#
#

#class ConfidencePenaltyLoss(Serializable):
#  """
#  The confidence penalty.
#  part of: https://arxiv.org/pdf/1701.06548.pdf
#
#  Calculate the -entropy for the given (batched policy).
#  Entropy is used as an additional loss so that it will penalize a too confident network.
#  """
#
#
#  @xnmt.serializable_init
#  def __init__(self, softmax): pass
#
#  def _perform_calc_loss(self,
#                         model: 'model_base.ConditionedModel',
#                         src: Union[sent.Sentence, 'batchers.Batch'],
#                         trg: Union[sent.Sentence, 'batchers.Batch']) -> losses.FactoredLossExpr:
#    neg_entropy = []
#    units = np.zeros(policy[0].dim()[1])
#    for i, ll in enumerate(policy):
#      if self.valid_pos[i] is not None:
#        ll = dy.pick_batch_elems(ll, self.valid_pos[i])
#        units[self.valid_pos[i]] += 1
#      else:
#        units += 1
#      loss = dy.sum_batches(dy.sum_elems(dy.cmult(dy.exp(ll), ll)))
#      neg_entropy.append(dy.sum_batches(loss))
#    return losses.LossExpr(dy.esum(neg_entropy), units)
