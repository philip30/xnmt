import numpy as np
import dynet as dy

from typing import List

import xnmt.batchers as batchers
import xnmt.transducers.base as transducers_base
import xnmt.transducers.recurrent as recurrent
import xnmt.rl.policy_network as network
import xnmt.rl.policy_priors as prior
import xnmt.persistence as persistence
import xnmt.expression_seqs as expr_seq
import xnmt.events as events
import xnmt.reports as reports
import xnmt.models.base as model_base

from xnmt.rl.policy_action import PolicyAction
from xnmt.persistence import bare
from xnmt import logger

class SegmentingSeqTransducer(transducers_base.SeqTransducer, persistence.Serializable,
                              reports.Reportable, model_base.PolicyConditionedModel):
  """
  A transducer that perform composition on smaller units (characters) into bigger units (words).
  This transducer will predict/use the segmentation discrete actions to compose the inputs.

  The composition function is defined by segment_composer. Please see segmenting_composer.py
  The final transducer is used to transduce the composed sequence. Usually it is a variant of RNN.

  ** USAGE
  This transducer is able to sample from several distribution.

  To segment from some predefined segmentations, please read the word corpus with CharFromWordTextReader.
  To learn the segmentation, please define the policy_learning. Please see rl/policy_learning.py
  To partly defined some segmentation using priors or gold input and learn from it, use the EpsilonGreedy with the proper priors. Please see priors.py.
  To sample from the policy instead doing argmax when doing inference please turn on the sample_during_search

  ** LEARNING
  By default it will use the policy gradient function to learn the network. The reward is composed by:

  REWARD = -sum(GENERATOR_LOSS) / len(TRG_SEQUENCE)

  Additional reward can be added by specifying the length prior. Please see length_prior.py

  ** REPORTING

  You can produce the predicted segmentation by using the SegmentationReporter in your inference configuration.
  This will produce one segmentation per line in {REPORT_PATH}.segment

  """
  yaml_tag = '!SegmentingSeqTransducer'

  @events.register_xnmt_handler
  @persistence.serializable_init
  def __init__(self,
               embed_encoder: transducers_base.SeqTransducer = bare(transducers_base.IdentitySeqTransducer),
               segment_composer: SequenceComposer = bare(SeqTransducerComposer),
               final_transducer: recurrent.BiLSTMSeqTransducer = bare(recurrent.BiLSTMSeqTransducer),
               policy_network: network.PolicyNetwork = None,
               policy_prior: prior.PolicyPrior = None,
               train_policy_oracle: bool=True,
               test_policy_oracle: bool=True):
    policy_network = self.add_serializable_component("policy_network", policy_network, lambda: policy_network)
    model_base.PolicyConditionedModel.__init__(policy_network, train_policy_oracle, test_policy_oracle)

    self.embed_encoder = self.add_serializable_component("embed_encoder", embed_encoder, lambda: embed_encoder)
    self.segment_composer = self.add_serializable_component("segment_composer", segment_composer, lambda: segment_composer)
    self.final_transducer = self.add_serializable_component("final_transducer", final_transducer, lambda: final_transducer)
    self.no_char_embed = issubclass(segment_composer.__class__, VocabBasedComposer)
    self.policy_prior = self.policy_prior

  def transduce(self, embed_sent: expr_seq.ExpressionSequence) -> List[expr_seq.ExpressionSequence]:
    self.create_trajectories(embed_sent, force_oracle=False)
    actions = [np.nonzero(a.content) for a in self.actions]
    actions = [[a for a in actions[i] if a < self.src_sents[i].len_unpadded()] for i in range(len(actions))]

    # Create sentence embedding
    outputs = []
    embeddings = dy.concatenate(embed_sent.expr_list, d=1)
    for i in range(self.src_sents.batch_size()):
      sequence = dy.pick_batch_elem(embeddings, i)
      src = self.src_sents[i]
      lower_bound = 0
      output = []
      for j, upper_bound in enumerate(actions[i]):
        char_sequence = dy.pick_range(sequence, lower_bound, upper_bound+1, 1) if self.no_char_embed else None
        output.append(self.segment_composer.compose_single(char_sequence, src, lower_bound, upper_bound+1))
        lower_bound = upper_bound+1
      outputs.append(output)


    outputs = pad_output()

    return self.final_transducer.transduce(outputs)

  def create_trajectories(self, src_embedding: expr_seq.ExpressionSequence, force_oracle=False):
    if len(self.actions) != 0:
      return

    from_oracle = self.policy_train_oracle if self.train else self.policy_test_oracle
    from_oracle = from_oracle or force_oracle

    if from_oracle:
      force_actions = [src.segment for src in self.src_sents]
    elif self.policy_prior is not None:
      force_actions = self.policy_prior.sample(self.src_sents.batch_size(), self.src_sents.sent_len())
    else:
      force_actions = [None] * self.src_sents.batch_size()

    if self.policy_network is None:
      sample = np.asarray(force_actions).transpose()
      for i in range(self.src_sents.sent_len()):
        self.actions.append(PolicyAction(sample[i]))
    else:
      for i in range(self.src_sents.sent_len()):
        mask = src_embedding.mask[i] if src_embedding.mask is not None else None
        policy_action = self.policy_network.sample_actions(src_embedding[i], force_actions[i], mask)
        self.actions.append(policy_action)

  def get_final_states(self) -> List[transducers_base.FinalTransducerState]:
    return self.final_transducer.get_final_states()

