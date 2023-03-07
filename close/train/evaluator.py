import re
from collections import defaultdict, Counter
from numbers import Number
from typing import Optional, List, Dict, Any
import numpy as np


from allennlp.common import FromParams, Registrable, Params
from dataclasses import dataclass, replace

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge

from l2v.data.coco_captioning import CaptioningExample
from l2v.data.visual_news import VisualNewsExample
from l2v.data.vqa_v2 import VqaExample
from l2v.eval.vqa_eval import vqa_preprocess
from l2v.model.model import ExampleOutput
from l2v.utils import py_utils
from l2v.utils.quiet_ptbtokenizer import QuitePTBTokenizer


class SubmissionFileBuilder(Registrable):
  def build(self, dataset, predictions, output_file):
    raise NotImplementedError()


@dataclass(frozen=True)
class ResultKey(FromParams):
  """Key for a result from a model"""

  metric_name: str
  subset_name: Optional[str] = None
  dataset_name: Optional[str] = None

  def __str__(self):
    out = [self.dataset_name, self.subset_name, self.metric_name]
    return "/".join(x for x in out if x is not None)

  def __repr__(self):
    return str(self)


class Evaluator(Registrable):
  """Computes evaluations metrics"""

  def evaluate(
      self, examples: List, predictions: Dict[str, Any],
      allow_partial=False, subset_mapping=None
  ) -> Dict[ResultKey, Number]:
    """Computes corpus wide metrics

    :param examples: List of source examples
    :param predictions: example key -> model output
    :param allow_partial: Allow the predictions to only cover a subset of `examples`,
                          in which only those predictions should be evaluated
    :param subset_mapping: Function that maps example -> list of strings, names of the subsets that
                           example is part of
    """
    raise NotImplementedError()


class PerExampleEvaluator(Evaluator):
  """Computes per-examples evaluations metrics"""

  def evaluate_examples(self, examples: List, predictions: Dict[str, Any])-> List[Dict[str, Number]]:
    raise NotImplementedError()

  def evaluate(
      self,
      examples: List,
      predictions: Dict[str, Any],
      allow_partial=False,
      mean=True,
      subset_mapping=None
  ) -> Dict[ResultKey, Number]:
    examples_with_predictions = [x for x in examples if x.get_example_id() in predictions]
    if not allow_partial and (len(examples) != len(examples_with_predictions)):
      raise ValueError(f"Only {len(examples_with_predictions)}/{len(examples)} "
                       f"of examples have predictions")
    examples = examples_with_predictions
    per_example_scores = self.evaluate_examples(examples, predictions)
    per_metric_scores = py_utils.transpose_list_of_dicts(per_example_scores)

    subsets = defaultdict(list)
    all_ids = [x.get_example_id() for x in examples]

    id_to_ix = {k: i for i, k in enumerate(all_ids)}
    subsets[None] = list(range(len(all_ids)))

    if subset_mapping is not None:
      for example in examples:
        example_id = id_to_ix[example.get_example_id()]
        for subset in subset_mapping(example):
          subsets[subset].append(example_id)

    out = {}
    for metric_name, score in per_metric_scores.items():
      score = np.array(score)
      for subset_name, ixs in subsets.items():
        if mean:
          out[ResultKey(metric_name, subset_name)] = float(np.mean(score[ixs]))
        else:
          out[ResultKey(metric_name, subset_name)] = (float(np.sum(score[ixs])), len(ixs))

    return out


@Evaluator.register("cap-evaluator")
class CaptionEvaluator(Evaluator):

  @classmethod
  def from_params(
      cls, params: Params, constructor_to_call=None,
      constructor_to_inspect=None, **extras
  ):
    if "per_caption" in params:
      del params["per_caption"]
    return super().from_params(params, constructor_to_call, constructor_to_inspect, **extras)

  def __init__(self, cider=True, bleu=4):
    self.cider = cider
    self.bleu = bleu
    scorers = {}
    if cider:
      # from exp.ours.eval.fast_cider import FastCider
      scorers["cider"] = Cider()
    if bleu:
      scorers["bleu"] = Bleu(bleu)
    self.scorers = scorers
    self.tokenizer = QuitePTBTokenizer()

  def evaluate(
      self,
      examples: List,
      predictions: Dict[str, Any],
      allow_partial=False,
      subset_mapping=None,
  ):
    examples_with_predictions = [x for x in examples if x.get_example_id() in predictions]
    if not allow_partial and (len(examples) != len(examples_with_predictions)):
      raise ValueError(f"Only {len(examples_with_predictions)}/{len(examples)} "
                       f"of examples have predictions")
    examples = examples_with_predictions

    subsets = defaultdict(list)
    subsets[None] = examples
    if subset_mapping is not None:
      for example in examples:
        example_subsets = subset_mapping(example)
        for subset in example_subsets:
          subsets[subset].append(example)

    out = {}
    for subset_name, examples in subsets.items():
      all_scores = self._get_scores(examples, predictions)

      results = {}
      for name, scorer in self.scorers.items():
        corpus_scores, _ = all_scores[name]
        if isinstance(scorer, Cider):
          results["cider"] = corpus_scores
        elif isinstance(scorer, Bleu):
          scores, _ = all_scores[name]
          for i, score in enumerate(corpus_scores):
            results[f"bleu{i+1}"] = score
      if subset_name is not None:
        results["n"] = len(examples)
      out.update({ResultKey(metric_name=k, subset_name=subset_name): v for k, v in results.items()})
    return out

  def evaluate_examples(self, examples: List[CaptioningExample], predictions: Dict[str, Any]):
    all_scores = self._get_scores(examples, predictions)

    per_examples_scores = [{} for _ in examples]
    for name, scorer in self.scorers.items():
      score, scores = all_scores[name]
      if isinstance(scorer, Cider):
        for score, ex_scores in zip(scores, per_examples_scores):
          ex_scores["cider"] = score
      elif isinstance(scorer, Bleu):
        scores = py_utils.transpose_lists(scores)
        for score, ex_scores in zip(scores, per_examples_scores):
          for i, s in enumerate(score):
            ex_scores[f"bleu{i+1}"] = s

    return per_examples_scores

  def _get_scores(self,  examples: List[CaptioningExample], predictions: Dict[str, Any]):
    gts = {}
    res = {}
    for ix, instance in enumerate(examples):
      key = instance.get_example_id()
      assert key not in res
      res[key] = [predictions[instance.get_example_id()].text[0]]
      gts[key] = [x.lower() for x in instance.captions]

    res = self.tokenizer.tokenize(res)
    gts = self.tokenizer.tokenize(gts)

    scores = {}
    for name, scorer in self.scorers.items():
      if isinstance(scorer, Bleu):
        scores[name] = scorer.compute_score(gts, res, verbose=0)
      else:
        scores[name] = scorer.compute_score(gts, res)
    return scores


def vqa_score(answer, ground_truth_answer_counts):
  normlized_answers = Counter()
  for k, v in ground_truth_answer_counts.items():
    normlized_answers[vqa_preprocess(k)] = v
  return min(normlized_answers.get(vqa_preprocess(answer), 0) / 3, 1)


@Evaluator.register("vqa-evaluator")
class VqaEvaluator(PerExampleEvaluator):

  def evaluate_examples(self, examples: List[VqaExample],
                        predictions: Dict[str, ExampleOutput], add_scores=False):
    out = []
    for example in examples:
      answer = predictions[example.example_id].text[0]
      score = vqa_score(answer, example.answers)
      out.append(dict(score=score))
    return out


@Evaluator.register("vis-news-evaluator")
class VisualNewsEvaluator(Evaluator):

  @classmethod
  def from_params(
      cls, params: Params, constructor_to_call=None,
      constructor_to_inspect=None, **extras
  ):
    return super().from_params(params, constructor_to_call, constructor_to_inspect, **extras)

  def __init__(self, cider=True, meteor=True, rouge=True, bleu=4):
    self.cider = cider
    self.meteor = meteor
    self.rouge = rouge
    self.bleu = bleu
    scorers = {}
    if cider:
      scorers["cider"] = Cider()
    if meteor:
      scorers["meteor"] = Meteor()
    if rouge:
      scorers["rouge"] = Rouge()
    if bleu:
      scorers["bleu"] = Bleu(bleu)
    self.scorers = scorers
    self.tokenizer = QuitePTBTokenizer()

  def evaluate(
      self,
      examples: List,
      predictions: Dict[str, Any],
      allow_partial=False,
      subset_mapping=None,
  ):
    examples_with_predictions = [x for x in examples if x.get_example_id() in predictions]
    if not allow_partial and (len(examples) != len(examples_with_predictions)):
      raise ValueError(f"Only {len(examples_with_predictions)}/{len(examples)} "
                       f"of examples have predictions")
    examples = examples_with_predictions

    subsets = defaultdict(list)
    subsets[None] = examples
    if subset_mapping is not None:
      for example in examples:
        example_subsets = subset_mapping(example)
        for subset in example_subsets:
          subsets[subset].append(example)

    out = {}
    for subset_name, examples in subsets.items():
      all_scores = self._get_scores(examples, predictions)

      results = {}
      for name, scorer in self.scorers.items():
        corpus_scores, _ = all_scores[name]
        if isinstance(scorer, Cider):
          results["cider"] = corpus_scores
        elif isinstance(scorer, Meteor):
          results["meteor"] = corpus_scores
        elif isinstance(scorer, Rouge):
          results["rouge"] = corpus_scores
        elif isinstance(scorer, Bleu):
          scores, _ = all_scores[name]
          for i, score in enumerate(corpus_scores):
            results[f"bleu{i+1}"] = score
      if subset_name is not None:
        results["n"] = len(examples)
      out.update({ResultKey(metric_name=k, subset_name=subset_name): v for k, v in results.items()})
    return out

  def evaluate_examples(self, examples: List[VisualNewsExample], predictions: Dict[str, Any]):
    all_scores = self._get_scores(examples, predictions)

    per_examples_scores = [{} for _ in examples]
    for name, scorer in self.scorers.items():
      score, scores = all_scores[name]
      if isinstance(scorer, Cider):
        for score, ex_scores in zip(scores, per_examples_scores):
          ex_scores["cider"] = score
      elif isinstance(scorer, Bleu):
        scores = py_utils.transpose_lists(scores)
        for score, ex_scores in zip(scores, per_examples_scores):
          for i, s in enumerate(score):
            ex_scores[f"bleu{i+1}"] = s

    return per_examples_scores

  def _get_scores(self,  examples: List[VisualNewsExample], predictions: Dict[str, Any]):
    MAX_LOG_EXAMPLES = 0 # adjust this to list more examples
    MAX_CAPTION_LEN = 1_800

    gts = {}
    res = {}
    for ix, instance in enumerate(examples):
      key = instance.get_example_id()
      assert key not in res
      res[key] = [predictions[instance.get_example_id()].text[0]]
      gts[key] = [instance.caption.lower()]

      if ix < MAX_LOG_EXAMPLES:
        print(f'example id: {instance.example_id}')
        print(f'image id: {instance.image_id}')
        print(f'news article: {instance.article[:MAX_CAPTION_LEN]}\n')
        print(f'target caption: {gts[key][0]}')
        print(f'predicted caption: {res[key][0]}\n')

    res = self.tokenizer.tokenize(res)
    gts = self.tokenizer.tokenize(gts)

    scores = {}
    for name, scorer in self.scorers.items():
      if isinstance(scorer, Bleu):
        scores[name] = scorer.compute_score(gts, res, verbose=0)
      else:
        scores[name] = scorer.compute_score(gts, res)
    return scores
