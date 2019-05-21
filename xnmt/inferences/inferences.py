import xnmt
import xnmt.models as models
import xnmt.inferences.search_strategies as search_strategies

from typing import Optional, Sequence, Union


class IndependentOutputInference(models.Inference, xnmt.Serializable):
  yaml_tag = "!IndependentOutputInference"
  """
  Inference when outputs are produced independently, including for classifiers that produce only a single output.

  Assumes that generator.generate() takes arguments src, idx

  Args:
    src_file: path of input src file to be translated
    trg_file: path of file where trg translatons will be written
    ref_file: path of file with reference translations, e.g. for forced decoding
    max_src_len: Remove sentences from data to decode that are longer than this on the source side
    max_num_sents: Stop decoding after the first n sentences.
    post_process: post-processing of translation outputs (available string shortcuts:  ``none``, ``join-char``,
                  ``join-bpe``, ``join-piece``)
    mode: type of decoding to perform.

          * ``onebest``: generate one best.
          * ``score``: output scores, useful for rescoring

    batcher: inference batcher, needed e.g. in connection with ``pad_src_token_to_multiple``
    reporter: a reporter to create reports for each decoded sentence
  """
  @xnmt.serializable_init
  def __init__(self,
               src_file: Optional[str] = None,
               trg_file: Optional[str] = None,
               ref_file: Optional[str] = None,
               max_src_len: Optional[int] = None,
               max_num_sents: Optional[int] = None,
               post_process: Optional[Union[str, models.OutputProcessor, Sequence[models.OutputProcessor]]] = None,
               batcher: xnmt.structs.InOrderBatcher = xnmt.bare(xnmt.structs.InOrderBatcher, batch_size=1),
               reporter: Optional[Union[models.Reporter, Sequence[models.Reporter]]] = None,
               loss_calculator: models.LossCalculator = xnmt.bare(xnmt.train.MLELoss)):
    super().__init__(src_file=src_file,
                     trg_file=trg_file,
                     ref_file=ref_file,
                     max_src_len=max_src_len,
                     max_num_sents=max_num_sents,
                     batcher=batcher,
                     reporter=reporter,
                     post_processor = xnmt.modules.output_processors.get_output_processor(post_process))
    self.loss_calculator = loss_calculator

  def generate_one(self, generator: models.GeneratorModel, src: xnmt.Batch) -> Sequence[xnmt.Sentence]:
    return generator.generate(src, None)

  def compute_losses_one(self, generator: models.ConditionedModel, src: xnmt.Batch, ref: xnmt.Batch) \
      -> xnmt.FactoredLossExpr:
    return self.loss_calculator.calc_loss(generator, src, ref)


class AutoRegressiveInference(models.Inference, xnmt.Serializable):
  yaml_tag = "!AutoRegressiveInference"
  """
  Performs inference for auto-regressive networks that expand based on their own previous outputs.

  Assumes that generator.generate() takes arguments src, idx, search_strategy, forced_trg_ids

  Args:
    src_file: path of input src file to be translated
    trg_file: path of file where trg translatons will be written
    ref_file: path of file with reference translations, e.g. for forced decoding
    max_src_len: Remove sentences from data to decode that are longer than this on the source side
    max_num_sents: Stop decoding after the first n sentences.
    post_process: post-processing of translation outputs
                  (available string shortcuts:  ``none``,``join-char``,``join-bpe``,``join-piece``)
    search_strategy: a search strategy used during decoding.
    mode: type of decoding to perform.

            * ``onebest``: generate one best.
            * ``score``: output scores, useful for rescoring
    batcher: inference batcher, needed e.g. in connection with ``pad_src_token_to_multiple``
    reporter: a reporter to create reports for each decoded sentence
  """
  @xnmt.serializable_init
  def __init__(self,
               src_file: Optional[str] = None,
               trg_file: Optional[str] = None,
               ref_file: Optional[str] = None,
               max_src_len: Optional[int] = None,
               max_num_sents: Optional[int] = None,
               post_process: Optional[Union[str, models.OutputProcessor, Sequence[models.OutputProcessor]]] = None,
               search_strategy: models.SearchStrategy = xnmt.bare(search_strategies.BeamSearch),
               batcher: xnmt.structs.InOrderBatcher = xnmt.bare(xnmt.structs.batchers.InOrderBatcher, batch_size=1),
               reporter: Optional[Union[models.Reporter, Sequence[models.Reporter]]] = None,
               loss_calculator: models.LossCalculator = xnmt.bare(xnmt.train.MLELoss)):
    super().__init__(src_file=src_file,
                     trg_file=trg_file,
                     ref_file=ref_file,
                     max_src_len=max_src_len,
                     max_num_sents=max_num_sents,
                     batcher=batcher,
                     reporter=reporter,
                     post_processor = xnmt.modules.output_processors.get_output_processor(post_process))

    self.search_strategy = search_strategy
    self.loss_calculator = loss_calculator

  def generate_one(self, generator: models.GeneratorModel, src: xnmt.Batch, ref: Optional[xnmt.Batch]) -> Sequence[xnmt.Sentence]:
    return generator.generate(src, search_strategy=self.search_strategy, ref=ref)

  def compute_losses_one(self, generator: models.ConditionedModel, src: xnmt.Batch, ref: xnmt.Batch) \
      -> xnmt.FactoredLossExpr:
    return self.loss_calculator.calc_loss(generator, src, ref)


