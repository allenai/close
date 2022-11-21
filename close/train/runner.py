import json
from collections import Callable
from dataclasses import dataclass
from typing import Dict, Union

import torch
from allennlp.common import FromParams, Params
from torch.utils.data import DataLoader
from tqdm import tqdm

from close.model.load_model import load_model
from close.model.model import ExampleOutput, Model, BeamSearchSpec
from close.utils import pytorch_utils, py_utils
from close.utils.to_params import to_params_any


def prediction_args_to_json(prediction_args):
  prediction_args_dict = {}
  for k, v in prediction_args.items():
    if isinstance(v, FromParams):
      v = to_params_any(v, Union[BeamSearchSpec, float, int, str])
    prediction_args_dict[k] = v
  return prediction_args_dict


def save_example_output(output: Dict[str, ExampleOutput], output_dir):
  predictions = {}
  for key, out in output.items():
    predictions[key] = dict(
      answer=out.text,
      probs=None if out.text_logprobs is None else out.text_logprobs.tolist()
    )

  with open(output_dir + "/predictions.json", "w") as f:
    json.dump(predictions, f)


@dataclass
class CollateWithBatch(Callable):
  collate: Callable

  def __call__(self, batch):
    return batch, self.collate(batch)


def build_per_example_output(examples, output, beams_to_keep=1):
  out = {}
  for ex, ex_out in zip(examples, output):
    out[ex.example_id] = ex_out.set_beams_to_keep(beams_to_keep)
  return out


def run(model, examples, device,
        batch_size, num_workers, prediction_args, beams_to_keep=1,
        desc="eval", nopbar=False):
  if len(set(ex.example_id for ex in examples)) != len(examples):
    raise ValueError("Repeated ids in examples")
  if isinstance(model, str):
    model = load_model(model, device=device)
  model.set_prediction_args(**prediction_args)

  loader = DataLoader(
    [model.preprocess_example(x) for x in examples],
    batch_size=batch_size,
    collate_fn=CollateWithBatch(model.get_collate()),
    num_workers=num_workers,
    shuffle=False,
    pin_memory=True
  )

  return run_model(model, loader, beams_to_keep, desc, nopbar,
                   prediction_args=prediction_args)


def run_model(
    model, data_loader, beams_to_keep=1,
    desc="eval", nopbar=False, model_device=None,
    prediction_args=None
) -> Dict[str, ExampleOutput]:
  if prediction_args is None:
    prediction_args = {}
  model.eval()
  if model_device is None:
    model_device = pytorch_utils.get_model_device(model)

  model.set_prediction_args(**prediction_args)

  if desc is None:
    desc = "eval"

  out = {}
  if nopbar:
    it = data_loader
  else:
    it = tqdm(data_loader, desc=desc, ncols=100)

  for examples, batch in it:
    batch = pytorch_utils.to_device(batch, model_device)
    with torch.no_grad():
      output = model.predict(**batch)

    out.update(build_per_example_output(examples, output, beams_to_keep))

  return out
