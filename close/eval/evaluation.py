import json
from datetime import datetime
from os.path import isdir
from typing import Dict, Any, Union

from close.model.model import ExampleOutput
from close.train.evaluator import Evaluator
from close.utils import py_utils
from close.utils.py_utils import load_json_object, dump_json_object
from close.utils.to_params import to_params


def save_evaluation(prefix_or_dir: str, evaluator: Evaluator, stats: Dict[str, Any]):
  """Save the results in `prefix_or_dir`

  :param prefix_or_dir: Where to save the results
  :param evaluator: Evaluator used, save for book-keeping purposes
  :param stats: The states to save
  """
  if isdir(prefix_or_dir) and not prefix_or_dir.endswith("/"):
    prefix_or_dir += "/"
  cache_file = prefix_or_dir + "eval.json"
  to_save = {("all" if k.subset_name is None else k.subset_name) + "/" + k.metric_name: v
             for k, v in stats.items()}
  to_save = dict(
    stats=to_save,
    evaluator=to_params(evaluator, Evaluator),
    date=datetime.now().strftime("%m%d-%H%M%S"),
    version=6,
  )
  dump_json_object(to_save, cache_file)


def save_predictions(predictions: Dict[str, Union[Dict, ExampleOutput]], output_dir):
  pred_dict = {}
  for key, pred in predictions.items():
    if isinstance(pred, ExampleOutput):
      pred_dict[key] = dict(
        text=pred.text,
        text_logprobs=pred.text_logprobs
      )
    else:
      pred_dict[key] = pred

  with open(output_dir + "/predictions.json", "w") as f:
    json.dump(pred_dict, f, cls=py_utils.NumpyArrayEncoder)


def load_predictions(file: str) -> Dict[str, ExampleOutput]:
  pred = load_json_object(file)
  return {k: ExampleOutput(v["text"], v["text_logprobs"]) for k, v in pred.items()}
