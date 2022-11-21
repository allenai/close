import re
from collections import defaultdict, Counter
from numbers import Number
from typing import Optional, List, Dict, Any
import numpy as np


from allennlp.common import FromParams, Registrable, Params
from dataclasses import dataclass, replace

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider

from close.data.coco_captioning import CaptioningExample
from close.data.visual_entailment import VisualEntailmentExample
from close.data.vqa_v2 import VqaExample
from close.eval.vqa_eval import vqa_preprocess
from close.model.model import ExampleOutput
from close.utils import py_utils
from close.utils.quiet_ptbtokenizer import QuitePTBTokenizer


class Evaluator(Registrable):
  """Computes evaluations metrics"""

  def evaluate(self, examples: List, predictions: Dict[str, Any]) -> Dict[str, Number]:
    """Computes corpus wide metrics

    :param examples: List of source examples
    :param predictions: example key -> model output
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
      mean=True,
  ) -> Dict[str, Number]:
    examples_with_predictions = [x for x in examples if x.example_id in predictions]
    if len(examples) != len(examples_with_predictions):
      raise ValueError(f"Only {len(examples_with_predictions)}/{len(examples)} "
                       f"of examples have predictions")
    examples = examples_with_predictions
    per_example_scores = self.evaluate_examples(examples, predictions)
    per_metric_scores = py_utils.transpose_list_of_dicts(per_example_scores)

    out = {}
    for metric_name, score in per_metric_scores.items():
      score = np.array(score)
      out[metric_name] = float(np.mean(score))

    return out


@Evaluator.register("cap-evaluator")
class CaptionEvaluator(Evaluator):

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
      subset_mapping=None,
  ):
    examples_with_predictions = [x for x in examples if x.example_id in predictions]
    if len(examples) != len(examples_with_predictions):
      raise ValueError(f"Only {len(examples_with_predictions)}/{len(examples)} "
                       f"of examples have predictions")
    examples = examples_with_predictions

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
    return results

  def _get_scores(self,  examples: List[CaptioningExample], predictions: Dict[str, Any]):
    gts = {}
    res = {}
    for ix, instance in enumerate(examples):
      key = instance.example_id
      assert key not in res
      res[key] = [predictions[instance.example_id].text[0]]
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


@Evaluator.register("entailment")
class EntailmentEvaluator(PerExampleEvaluator):

  def evaluate_examples(self, examples: List[VisualEntailmentExample], predictions: Dict[str, Any]):
    scores = []
    for ex in examples:
      text = predictions[ex.example_id].text[0]
      scores.append(dict(accuracy=text==ex.label))
    return scores
