"""
Reports gather inputs, outputs, and intermediate computations in a nicely formatted way for convenient manual inspection.

To support reporting, the networks providing the data to be reported must subclass ``Reportable`` and call
``self.report_sent_info(d)`` with key/value pairs containing the data to be reported at the appropriate times.
If this causes a computational overhead, the boolean ``compute_report`` field should queried and extra computations
skipped if this field is ``False``.

Next, a ``Reporter`` needs to be specified that supports reports based on the previously created key/value pairs.
Reporters are passed to inference classes, so it's possible e.g. to report only at the final test decoding, or specify
a special reporting inference object that only looks at a handful of xnmt, etc.

Note that currently reporting is only supported at test-time, not at training time.
"""

import os
import math
import numpy as np

from typing import  Optional, Sequence, Union
from collections import defaultdict
from bs4 import BeautifulSoup as bs4

import sys

import xnmt
import xnmt.models as models



class ProbReporter(models.Reporter, xnmt.Serializable):
  yaml_tag = "!ProbReporter"

  @xnmt.serializable_init
  def __init__(self, report_path: Optional[str] = None):
    self.report_path = report_path
    self.fp = None

  def create_sent_report(self, hyp: models.Hypothesis, src: xnmt.Sentence, hyp_num: int):
    if self.fp is None:
      self.fp = sys.stdout if self.report_path is None else open(self.report_path, "w")

    log_prob = []
    now = hyp
    while now.parent is not None:
      log_prob.append(now.action.log_likelihood.scalar_value())
      now = now.parent

    print(" ".join(map(lambda x: f"{x:.5f}", list(reversed(np.exp(log_prob))))), file=self.fp)

  def conclude_report(self):
    if self.fp is not None and self.report_path is not None:
      self.fp.close()


#
#class ReferenceDiffReporter(models.Reporter, xnmt.Serializable):
#  yaml_tag = "!ReferenceDiffReporter"
#  """
#  Reporter that uses the CharCut tool for nicely displayed difference highlighting between outputs and references.
#
#  The stand-alone tool can be found at https://github.com/alardill/CharCut
#
#  Args:
#    match_size: min match size in characters (set < 3 e.g. for Japanese or Chinese)
#    alt_norm: alternative normalization scheme: use only the candidate's length for normalization
#    report_path: Path of directory to write HTML files to
#  """
#  @xnmt.serializable_init
#  @xnmt.register_xnmt_handler
#  def __init__(self,
#               match_size: int = 3,
#               alt_norm: bool = False,
#               report_path: str = xnmt.settings.DEFAULT_REPORT_PATH) -> None:
#    self.match_size = match_size
#    self.alt_norm = alt_norm
#    self.report_path = report_path
#    self.hyp_sents, self.ref_sents, self.src_sents = [], [], []
#
#  def create_sent_report(self,
#                         src: xnmt.Sentence,
#                         output: xnmt.structs.sentences.ReadableSentence,
#                         ref_file: Optional[str] = None,
#                         **kwargs) -> None:
#    """
#    Create report.
#
#    Args:
#      src: source-side input
#      output: generated output
#      ref_file: path to reference file
#      **kwargs: arguments to be ignored
#    """
#    reference = xnmt.utils.cached_file_lines(ref_file)[output.idx]
#    trg_str = output.sent_str()
#    if isinstance(src, xnmt.structs.sentences.ReadableSentence):
#      src_str = src.sent_str()
#      self.src_sents.append(src_str)
#    self.hyp_sents.append(trg_str)
#    self.ref_sents.append(reference)
#
#  def conclude_report(self) -> None:
#    if self.hyp_sents:
#      html_filename = os.path.join(self.report_path, "charcut.html")
#      xnmt.utils.make_parent_dir(html_filename)
#      args = xnmt.utils.ArgClass(html_output_file=html_filename, match_size=self.match_size, alt_norm=self.alt_norm)
#      aligned_segs = charcut.load_input_segs(cand_segs=self.hyp_sents,
#                                             ref_segs=self.ref_sents,
#                                             src_segs=self.src_sents)
#      charcut.run_on(aligned_segs, args)
#      self.hyp_sents, self.ref_sents, self.src_sents = [], [], []
#
#
#class CompareMtReporter(models.Reporter, xnmt.Serializable):
#  yaml_tag = "!CompareMtReporter"
#  """
#  Reporter that uses the compare-mt.py script to analyze and compare MT results.
#
#  The stand-alone tool can be found at https://github.com/neubig/util-scripts
#
#  Args:
#    out2_file: A path to another system output. Add only if you want to compare outputs from two systems.
#    train_file: A link to the training corpus target file
#    train_counts: A link to the training word frequency counts as a tab-separated "word\\tfreq" file
#    alpha: A smoothing coefficient to control how much the model focuses on low- and high-frequency events.
#           1.0 should be fine most of the time.
#    ngram: Maximum length of n-grams.
#    sent_size: How many xnmt to print.
#    ngram_size: How many n-grams to print.
#
#    report_path: Path of directory to write report files to
#  """
#  yaml_tag = "!CompareMtReporter"
#  @xnmt.serializable_init
#  @xnmt.register_xnmt_handler
#  def __init__(self,
#               out2_file: Optional[str] = None,
#               train_file: Optional[str] = None,
#               train_counts: Optional[str] = None,
#               alpha: float = 1.0,
#               ngram: int = 4,
#               ngram_size: int = 50,
#               sent_size: int = 10,
#               report_path: str = xnmt.settings.DEFAULT_REPORT_PATH) -> None:
#    self.out2_file = out2_file
#    self.train_file = train_file
#    self.train_counts = train_counts
#    self.alpha = alpha
#    self.ngram = ngram
#    self.ngram_size = ngram_size
#    self.sent_size = sent_size
#    self.report_path = report_path
#    self.hyp_sents, self.ref_sents = [], []
#    self.src_sents = []
#
#  def create_sent_report(self, output: xnmt.structs.sentences.ReadableSentence, ref_file: str, **kwargs) -> None:
#    """
#    Create report.
#
#    Args:
#      output: generated output
#      ref_file: path to reference file
#      **kwargs: arguments to be ignored
#    """
#    reference = xnmt.utils.cached_file_lines(ref_file)[output.idx]
#    trg_str = output.sent_str()
#    self.hyp_sents.append(trg_str)
#    self.ref_sents.append(reference)
#
#  def conclude_report(self) -> None:
#    if self.hyp_sents:
#      ref_filename = os.path.join(self.report_path, "tmp", "compare-mt.ref")
#      out_filename = os.path.join(self.report_path, "tmp", "compare-mt.out")
#      xnmt.utils.make_parent_dir(out_filename)
#      with open(ref_filename, "w") as fout:
#        for l in self.ref_sents: fout.write(f"{l.strip()}\n")
#      with open(out_filename, "w") as fout:
#        for l in self.hyp_sents: fout.write(f"{l.strip()}\n")
#      import xnmt.thirdparty.comparemt.compare_mt as compare_mt
#      args = xnmt.utils.ArgClass(ref_file = ref_filename,
#                            out_file = out_filename,
#                            out2_file = self.out2_file,
#                            train_file = self.train_file,
#                            train_counts = self.train_counts,
#                            alpha = self.alpha,
#                            ngram = self.ngram,
#                            ngram_size = self.ngram_size,
#                            sent_size = self.sent_size)
#      out_lines = compare_mt.main(args)
#      report_filename = os.path.join(self.report_path, "compare-mt.txt")
#      xnmt.utils.make_parent_dir(report_filename)
#      with open(report_filename, "w") as fout:
#        for l in out_lines: fout.write(f"{l}\n")
#      self.hyp_sents, self.ref_sents, self.src_sents = [], [], []
#
#
#class HtmlReporter(models.Reporter):
#  """
#  A base class for reporters that produce HTML outputs that takes care of some common functionality.
#
#  Args:
#    report_name: prefix for report files
#    report_path: Path of directory to write HTML and image files to
#  """
#  def __init__(self, report_name: str, report_path: str = xnmt.settings.DEFAULT_REPORT_PATH) -> None:
#    self.report_name = report_name
#    self.report_path = report_path
#    self.html_contents = ["<html><meta charset='UTF-8' /><head><title>Translation Report</title></head><body>"]
#    self.html_contents.append("""
#      <style>
#        body {font-family: sans-serif; font-size: 11pt;}
#        table, td, th {border-spacing: 0;}
#        th {padding: 10px;}
#        td {padding: 5px;}
#        th {border-top: solid black 2px; font-weight: normal;}
#        .tophead {border-bottom: solid black 1px;}
#        .src {font-style: oblique;}
#        .trg {font-family: Consolas, monospace;}
#        .del {font-weight: bold; color: #f00000;}
#        .ins {font-weight: bold; color: #0040ff;}
#        .shift {font-weight: bold;}
#        .match {}
#        .mainrow {border-top: solid black 1px; padding: 1em;}
#        .midrow {border-bottom: dotted gray 1px;}
#        .seghead {color: gray; text-align: right;}
#        .score {font-family: Consolas, monospace; text-align: right; font-size: large;}
#        .detail {font-size: xx-small; color: gray;}
#      </style>
#      <script>
#        function enter(cls) {
#          var elts = document.getElementsByClassName(cls);
#          for (var i=0; i<elts.length; i++)
#            elts[i].style.backgroundColor = "yellow";
#        }
#        function leave(cls) {
#          var elts = document.getElementsByClassName(cls);
#          for (var i=0; i<elts.length; i++)
#            elts[i].style.backgroundColor = "transparent";
#        }
#      </script>
#   """)
#
#  def add_sent_heading(self, idx: int) -> None:
#    self.html_contents.append(f"<h1>Translation Report for Sentence {idx}</h1>")
#    self.html_contents.append("<table>")
#
#  def finish_sent(self) -> None:
#    self.html_contents.append("</table>")
#
#  def finish_html_doc(self) -> None:
#    self.html_contents.append("</body></html>")
#
#  def write_html(self) -> None:
#    html_str = "\n".join(self.html_contents)
#    soup = bs4(html_str, "lxml")
#    pretty_html = soup.prettify()
#    html_file_name = os.path.join(self.report_path, f"{self.report_name}.html")
#    xnmt.utils.make_parent_dir(html_file_name)
#    with open(html_file_name, 'w', encoding='utf-8') as f:
#      f.write(pretty_html)
#
#  def add_fields_if_set(self, fields: dict) -> None:
#    html_ret = ""
#    for key, val in fields.items():
#      if val:
#        html_ret += f"<tr><td class='seghead'>{key}:</td><td>{val}</td></tr>"
#    if html_ret:
#      self.html_contents.append(html_ret)
#
#  def add_charcut_diff(self,
#                       trg_str: str,
#                       reference: str,
#                       match_size: int=3,
#                       alt_norm: bool = False,
#                       mt_label: str  = "MT:",
#                       ref_label: str  = "Ref:") -> None:
#    aligned_segs = charcut.load_input_segs(cand_segs=[trg_str],
#                                           ref_segs=[reference])
#    styled_ops = [charcut.compare_segments(cand, ref, match_size)
#                  for seg_id, _, _, cand, ref in aligned_segs]
#
#    seg_scores = list(charcut.score_all(aligned_segs, styled_ops, alt_norm))
#    # doc_cost = sum(cost for cost, _ in seg_scores)
#    # doc_div = sum(div for _, div in seg_scores)
#
#    self.html_contents.append(charcut.segs2html(aligned_segs[0], styled_ops[0], seg_scores[0], mt_label=mt_label,
#                                                ref_label=ref_label, use_id_col=False))
#
#
#class AttentionReporter(HtmlReporter, xnmt.Serializable):
#  yaml_tag = "!AttentionReporter"
#  """
#  Reporter that writes attention matrices to HTML.
#
#  Args:
#    max_num_sents: create attention report for only the first n xnmt
#    report_name: prefix for output files
#    report_path: Path of directory to write HTML and image files to
#  """
#
#  yaml_tag = "!AttentionReporter"
#
#  @xnmt.register_xnmt_handler
#  @xnmt.serializable_init
#  def __init__(self,
#               max_num_sents: Optional[int] = 100,
#               report_name: str = "attention",
#               report_path: str = xnmt.settings.DEFAULT_REPORT_PATH) -> None:
#    super().__init__(report_name=report_name, report_path=report_path)
#    self.max_num_sents = max_num_sents
#    self.cur_sent_no = 0
#
#
#  def create_sent_report(self, src: xnmt.Sentence, output: xnmt.structs.sentences.ReadableSentence, attentions: np.ndarray,
#                         ref_file: Optional[str], **kwargs) -> None:
#
#    """
#    Create report.
#
#    Args:
#      src: source-side input
#      output: generated output
#      attentions: attention matrices
#      ref_file: path to reference file
#      **kwargs: arguments to be ignored
#    """
#    self.cur_sent_no += 1
#    if self.max_num_sents and self.cur_sent_no > self.max_num_sents: return
#    reference = xnmt.utils.cached_file_lines(ref_file)[output.idx]
#    idx = src.idx
#    self.add_sent_heading(idx)
#    src_tokens = src.str_tokens() if isinstance(src, xnmt.structs.sentences.ReadableSentence) else []
#    trg_tokens = output.str_tokens()
#    src_str = src.sent_str() if isinstance(src, xnmt.structs.sentences.ReadableSentence) else ""
#    trg_str = output.sent_str()
#    self.add_charcut_diff(trg_str, reference)
#    self.add_fields_if_set({"Src" : src_str})
#    self.add_atts(attentions, src.get_array() if isinstance(src, xnmt.structs.sentences.ArraySentence) else src_tokens,
#                  trg_tokens, idx)
#    self.finish_sent()
#
#  def conclude_report(self) -> None:
#    self.finish_html_doc()
#    self.write_html()
#    self.cur_sent_no = 0
#
#  def add_atts(self,
#               attentions: np.ndarray,
#               src_tokens: Union[Sequence[str], np.ndarray],
#               trg_tokens: Sequence[str],
#               idx: int,
#               desc: str = "Attentions") -> None:
#    """
#    Add attention matrix to HTML code.
#
#    Args:
#      attentions: numpy array of dimensions (src_len x trg_len)
#      src_tokens: list of strings (case of src text) or numpy array of dims (nfeat x speech_len) (case of src speech)
#      trg_tokens: list of string tokens
#      idx: sentence no
#      desc: readable description
#    """
#    src_is_speech = isinstance(src_tokens, np.ndarray)
#    size_x = math.log(len(trg_tokens)+2) * 3
#    if src_is_speech:
#      size_y = math.log(src_tokens.shape[1]+2)
#    else:
#      size_y = math.log(len(src_tokens)+2) * 3
#    attention_file = f"{self.report_path}/img/attention.{xnmt.utils.valid_filename(desc).lower()}.{idx}.png"
#    html_att = f'<tr><td class="seghead">{desc}:</td><td></td></tr>' \
#               f'<tr><td colspan="2" align="left"><img src="img/{os.path.basename(attention_file)}" alt="attention matrix" /></td></tr>'
#    plotting.plot_attention(src_words=src_tokens, trg_words=trg_tokens, attention_matrix=attentions,
#                            file_name=attention_file, size_x=size_x, size_y=size_y)
#    self.html_contents.append(html_att)
#
#
#class OOVStatisticsReporter(models.Reporter, xnmt.Serializable):
#  yaml_tag = "!OOVStatisticsReporter"
#  """
#  A reporter that prints OOV statistics: recovered OOVs, fantasized new words, etc.
#
#  Some networks such as character- or subword-based networks can produce words that are not in the training.
#  This is desirable when we produce a correct word that would have been an OOV with a word-based model
#  but undesirable when we produce something that's not a correct word.
#  The reporter prints some statistics that help analyze the OOV behavior of the model.
#
#  Args:
#    train_trg_file: path to word-tokenized training target file
#    report_path: Path of directory to write text files to
#  """
#  yaml_tag = "!OOVStatisticsReporter"
#
#  @xnmt.serializable_init
#  @xnmt.register_xnmt_handler
#  def __init__(self, train_trg_file: str, report_path: str = xnmt.settings.DEFAULT_REPORT_PATH,
#               trg_reader: models.InputReader = xnmt.ref_trg_reader) -> None:
#    self.report_path = report_path
#    self.report_fp = None
#    self.train_trg_file = train_trg_file
#    self.out_sents, self.ref_lines = [], []
#    if isinstance(trg_reader, xnmt.modules.input_readers.BaseTextReader):
#      self.output_vocab = trg_reader.vocab
#
#  def create_sent_report(self, output: xnmt.structs.sentences.ReadableSentence, ref_file: str, **kwargs) -> None:
#    reference = xnmt.utils.cached_file_lines(ref_file)[output.idx]
#    self.ref_lines.append(reference)
#    self.out_sents.append(output)
#
#  def conclude_report(self) -> None:
#    train_words = set()
#    with open(self.train_trg_file) as f_train:
#      for line in f_train:
#        for word in line.strip().split():
#          train_words.add(word)
#    ref_words = set()
#    ref_words_oov = defaultdict(int)
#    ref_words_total = 0
#    ref_oovs_unmatched = defaultdict(int)
#    for ref_line, trg_sent in zip(self.ref_lines, self.out_sents):
#      for word in ref_line.strip().split():
#        ref_words.add(word)
#        ref_words_total += 1
#        if word not in train_words:
#          ref_words_oov[word] += 1
#          if word not in trg_sent.sent_str().split():
#            ref_oovs_unmatched[word] += 1
#    if ref_words_total == 0: raise ValueError("Found empty reference")
#    hyp_words = set()
#    hyp_words_oov = defaultdict(int)
#    hyp_words_total = 0
#    hyp_oovs_matched = defaultdict(int)
#    hyp_oovs_unmatched = defaultdict(int)
#    for trg_sent, ref_line in zip(self.out_sents, self.ref_lines):
#      for word in trg_sent.sent_str().split():
#        hyp_words.add(word)
#        hyp_words_total += 1
#        if word not in train_words:
#          hyp_words_oov[word] += 1
#          ref_line_words = ref_line.strip().split()
#          if word in ref_line_words:
#            hyp_oovs_matched[word] += 1
#          else:
#            hyp_oovs_unmatched[word] += 1
#    if hyp_words_total == 0: raise ValueError("Found empty hypothesis")
#    sorted_hyp_oovs_matched = sorted(hyp_oovs_matched.items(), key=lambda x: x[1], reverse=True)
#    sorted_hyp_oovs_unmatched = sorted(hyp_oovs_unmatched.items(), key=lambda x: x[1], reverse=True)
#    sorted_ref_oovs_unmatched = sorted(ref_oovs_unmatched.items(), key=lambda x: x[1], reverse=True)
#    num_oovs_ref = sum(ref_words_oov.values())
#    num_oovs_hyp = sum(hyp_words_oov.values())
#    num_oovs_hyp_matched = sum(hyp_oovs_matched.values())
#    hyp_oov_prec = f"{num_oovs_hyp_matched/num_oovs_hyp*100:.2f}%" if num_oovs_hyp>0 else "n/a"
#    hyp_oov_rec = f"{num_oovs_hyp_matched/num_oovs_ref*100:.2f}%" if num_oovs_ref>0 else "n/a"
#    with open(os.path.join(self.report_path, "oov-statistics.txt"), "w") as fout:
#      fout.write(f"OOV Statistics Report\n---------------------\n")
#      fout.write(f"Size of subword vocab:                      {len(self.output_vocab)}\n")
#      fout.write(f"Word types in training corpus:              {len(train_words)}\n")
#      fout.write(f"Word types in test reference:               {len(ref_words)}\n")
#      fout.write(f"Word types in test hypothesis:              {len(hyp_words)}\n")
#      fout.write(f"OOV words in test reference:                {num_oovs_ref} ({num_oovs_ref/ref_words_total*100:.2f}%)\n")
#      fout.write(f"OOV words in test hypothesis:               {num_oovs_hyp} ({num_oovs_hyp/hyp_words_total*100:.2f}%)\n")
#      fout.write(f"Hypothesis OOVs precision (sentence-match): {hyp_oov_prec}\n")
#      fout.write(f"Hypothesis OOVs recall (sentence-match):    {hyp_oov_rec}\n")
#      fout.write(f"\n\nListing:\n")
#      fout.write(f"OOVs recovered: \n")
#      fout.write("\n".join([f"{item[0]} ({item[1]})" for item in sorted_hyp_oovs_matched]))
#      fout.write(f"\n\nOOVs not recovered: \n")
#      fout.write("\n".join([f"{item[0]} ({item[1]})" for item in sorted_ref_oovs_unmatched]))
#      fout.write(f"\n\nfantasized words: \n")
#      fout.write("\n".join([f"{item[0]} ({item[1]})" for item in sorted_hyp_oovs_unmatched]))
#
