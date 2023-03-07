import argparse
import json
import logging
import os
from typing import Union

import numpy as np

from l2v.data.coco_captioning import CocoCaptioning, CocoSCE
from l2v.data.dataset import Dataset
from l2v.data.visual_news import VisualNews
from l2v.data.vqa_e import EVQA
from l2v.data.vqa_v2 import Vqa2
from l2v.eval.evaluation import save_predictions, save_evaluation
from l2v.model.model import BeamSearchSpec
from l2v.train.evaluator import CaptionEvaluator, Evaluator, ResultKey, VqaEvaluator, VisualNewsEvaluator
from l2v.train.runner import prediction_args_to_json, run
from l2v.utils import py_utils, pytorch_utils
from l2v.utils.to_params import to_params

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datetime import datetime
from os.path import join, exists, dirname
from shutil import rmtree


def get_default_seq_len(ds: Dataset) -> int:
  if isinstance(ds, (CocoCaptioning, CocoSCE)):
    return 30
  if isinstance(ds, (Vqa2, EVQA)):
    return 24
  if isinstance(ds, (VisualNews)):
    return 30
  else:
    raise NotImplementedError(f"No default lengths set for dataset {ds}")


def get_evaluator(ds: Dataset) -> Union[Evaluator, None]:
  if isinstance(ds, (CocoCaptioning, CocoSCE)):
    return CaptionEvaluator()
  if isinstance(ds, (Vqa2, EVQA)):
    return VqaEvaluator()
  if isinstance(ds, (VisualNews)):
    return VisualNewsEvaluator()
  else:
    raise ValueError()


def eval_on(args, run_dir, dataset, evaluator, prediction_args, devices, skip_existing=True):
  if args.output_dir:
    output_dir = args.output_dir

  elif args.output_name is not None:
    if args.output_name == "":
      name = f"{dataset.get_name()}"
    else:
      name = f"{dataset.get_name()}--{args.output_name}"
    eval_dir = join(run_dir, "eval")
    if not exists(eval_dir):
      os.mkdir(eval_dir)
    output_dir = join(eval_dir, name)
  else:
    output_dir = None

  if output_dir is not None:
    if exists(output_dir):
      if len(os.listdir(output_dir)) > 0:
        if skip_existing:
          logging.info(f"{output_dir} already exists, skipping")
          return

        if args.override or py_utils.get_yes_no(f"{output_dir} exists, delete (y/n)?"):
          logging.info(f"Deleting {output_dir}")
          rmtree(output_dir)
        else:
          logging.info("No override, not stopping")
          return
    elif not exists(dirname(output_dir)):
      raise ValueError(f"Parent folder {dirname(output_dir)} does not exist")
    else:
      logging.info(f"Will save to {output_dir}")
  else:
    logging.info(f"Not saving the output")

  if output_dir:
    if not exists(output_dir):
      os.mkdir(output_dir)
    logging.info(f"Saving output to {output_dir}")

  logging.info("Setting up...")
  examples = dataset.load()

  if args.dry_run:
    logging.info("Skipping running the model since this is a dry run")
    return

  beams_to_keep = vars(args).get("beams_to_keep")
  batch_size = args.batch_size

  output = run(
    run_dir, examples, devices, batch_size, args.num_workers,
    prediction_args, beams_to_keep=beams_to_keep)

  if output_dir is not None:
    logging.info(f"Saving output to {output_dir}")
    save_predictions(output, output_dir)

    config = dict(
      batch_size=batch_size,
      num_workers=args.num_workers,
      predictions_args=prediction_args_to_json(prediction_args),
      dataset=to_params(dataset, Dataset),
      beams_to_keep=beams_to_keep,
      date=datetime.now().strftime("%m%d-%H%M%S"),
    )

    with open(output_dir + "/config.json", "w") as f:
      json.dump(config, f, indent=2)

  logging.info("Evaluating...")
  if isinstance(evaluator, Evaluator):
    results = evaluator.evaluate(examples, output, allow_partial=True, subset_mapping=None)
    k = [k for k in results if k.metric_name == "n"]
    if len(k) == 1:
      del results[k[0]]

    if output_dir is not None:
      results[ResultKey("n", None)] = len(output)
      logging.info(f"Caching evaluation to {output_dir}")
      save_evaluation(output_dir, evaluator, results)

    results = {str(k): v for k, v in results.items()}
    print(json.dumps(results, indent=2))

  elif evaluator is None:
    logging.info(f"No evaluator for this data")

  elif output_dir is not None:
    submission_file = join(output_dir, "submission.json")
    logging.info(f"Building submission file {submission_file}")
    evaluator.build(dataset, output, submission_file)


def eval_generative_model(args, run_dir, dataset, devices, skip_existing=True):
  prediction_args = {}
  arg_dict = vars(args)
  prediction_args["test_on_l"] = arg_dict.get("test_on_l", False)
  if "image_cache" in arg_dict:
    prediction_args["image_cache"] = arg_dict["image_cache"]

  if arg_dict.get("max_seq_len"):
    max_seq_len = arg_dict["max_seq_len"]
  else:
    max_seq_len = get_default_seq_len(dataset)
    logging.info(f"Defaulting to max_seq_len {max_seq_len} for dataset {dataset.get_name()}")

  if max_seq_len is not None:
    bs = BeamSearchSpec(beam_size=args.beam_size, max_seq_len=max_seq_len)
  else:
    bs = None
  prediction_args["beam_search_spec"] = bs
  evaluator = get_evaluator(dataset)
  eval_on(args, run_dir, dataset, evaluator, prediction_args, devices, skip_existing)


def main():
  parser = argparse.ArgumentParser(description="Compute predictions for a GPV model")
  parser.add_argument("model")
  parser.add_argument("--data", default=["coco"], nargs="+")
  parser.add_argument("--image_cache", default=None)
  parser.add_argument("--device", nargs="+", default=[None], help="GPU devices to use")
  parser.add_argument("--batch_size", type=int, default=30)
  parser.add_argument("--num_workers", type=int, default=4)
  parser.add_argument("--sample", type=int, default=None)
  parser.add_argument("--beams_to_keep", type=int, default=5, help="Number of predictions to save")
  parser.add_argument("--max_seq_len", type=int, default=None)
  parser.add_argument("--beam_size", type=int, default=5)
  parser.add_argument("--test_on_l", action="store_true")
  parser.add_argument("--noeval", action="store_true", help="Evaluate the results")
  parser.add_argument("--override", action="store_true", help="Delete output dir if it exists")
  parser.add_argument("--output_dir", help="Save to this directory")
  parser.add_argument("--output_name",
                      help="Save results in model/run/eval/{dataset_name}--{output_name}")
  parser.add_argument("--dry_run", action="store_true")
  args = parser.parse_args()

  py_utils.add_stdout_logger()

  if args.output_dir and args.output_name:
    raise ValueError("Cannot specify output_name and output_dir")

  models = py_utils.find_models(args.model, require_done=False)
  if len(models) == 0:
    logging.info("No models selected")
    return

  datasets = []
  for ds in args.data:
    if ds == "coco":
      datasets.append(CocoCaptioning("val", sample=args.sample))
    elif ds == "evqa":
      datasets.append(EVQA("val", sample=args.sample))
    elif ds == "vis-news":
      datasets.append(VisualNews("test", sample=args.sample))
    else:
      raise RuntimeError(ds)

  devices = pytorch_utils.get_devices(args.device)
  if args.output_dir:
    models = py_utils.flatten_list(x[1] for x in models.values())
    if len(models) > 1:
      raise ValueError("Cannot use one output dir if more than one model selected!")
    model = models[0]

    if len(datasets) > 1:
      raise ValueError("Cannot use one output dir if more than one dataset is selected!")
    if len(datasets) == 0:
      raise ValueError("No datasets is selected!")
    eval_generative_model(args, model, datasets[0], devices, skip_existing=False)

  else:
    targets = []
    for model_name, (model_dir, runs) in models.items():
      for ds in datasets:
        for run_dir in runs:
          targets.append((run_dir, ds))

    if len(targets) == 0:
      raise ValueError("No datasets to evaluate on found!")

    for i, (run_dir, dataset) in enumerate(targets):
      if len(targets) > 1:
        logging.info(f"Evaluating on {run_dir} {dataset.get_name()} ({i+1}/{len(targets)})")
      else:
        logging.info(f"Evaluating on {run_dir} {dataset.get_name()}")

      eval_generative_model(args, run_dir, dataset, devices, skip_existing=len(targets) > 1)


if __name__ == '__main__':
  main()
