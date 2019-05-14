

class SearchOutputProcessor(object):
  def process_output(self, search_output):
    pass

class TranslatorProcessor(object):
  pass


#    return sent.SimpleSentence(idx=src[0].idx,
#                               words=output_actions,
#                               vocab=getattr(self.trg_reader, "vocab", None),
#                               output_procs=self.trg_reader.output_procs,
#                               score=score)

#    return sent.DepTreeRNNGSequenceSentence(idx=src[0].idx,
#                                            score=score,
#                                            actions=output_actions,
#                                            surface_vocab=getattr(self.trg_reader, "surface_vocab", None),
#                                            nt_vocab=getattr(self.trg_reader, "nt_vocab", None),
#                                            edge_vocab=getattr(self.trg_reader, "edge_vocab", None),
#                                            output_procs=self.trg_reader.output_procs)


# outputs = []
#    for curr_output in sorted_outputs:
#      output_actions = [x for x in curr_output.word_ids[0]]
#      attentions = [x for x in curr_output.attentions[0]]
#      score = curr_output.score[0]
#      out_sent = self._emit_translation(src, output_actions, score)
#      if len(sorted_outputs) == 1:
#        outputs.append(out_sent)
#      else:
#        outputs.append(sent.NbestSentence(base_sent=out_sent, nbest_id=src[0].idx))
#
#    if self.is_reporting():
#      attentions = np.concatenate([x.npvalue() for x in attentions], axis=1)
#      self.report_sent_info({"attentions": attentions,
#                             "src": src[0],
#                             "output": outputs[0]})
#
#    return outputs
