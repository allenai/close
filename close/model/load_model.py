import logging
from os import listdir
from os.path import dirname, join, exists

import torch
from allennlp.common import Params

from close.model.model import Model, BEST_STATE_NAME
from close.utils import py_utils
from close.utils.py_utils import load_json_object, import_all, select_run_dir


def load_model(run_dir, use_best_weights=True, device=None, epoch=None):
  import_all()
  if run_dir.endswith("/"):
    run_dir = run_dir[:-1]

  run_dir = select_run_dir(run_dir)

  model_spec = join(dirname(run_dir), "model.json")
  params = Params(load_json_object(model_spec))

  with py_utils.DisableLogging():
    model: Model = Model.from_params(params)

  model.initialize(load_params=False)

  if epoch:
    src = join(run_dir, f"state-ep{epoch}.pth")
    if not exists(src):
      raise ValueError(f"Requested epoch {epoch} not found in {run_dir}")
    state_dict = torch.load(src, map_location="cpu")
  else:
    state_dict = None
    if use_best_weights:
      src = join(run_dir, BEST_STATE_NAME)
      if exists(src):
        state_dict = torch.load(src, map_location="cpu")
      else:
        logging.info(f"No best-path found for {run_dir}, using last saved state")

    if state_dict is None:
      epochs = [x for x in listdir(run_dir) if x.startswith("state-ep")]
      epochs.sort(key=lambda x: int(x.split(".pth")[0][len("state-ep"):]), reverse=True)
      if len(epochs) == 0:
        logging.info()
        raise ValueError(f"No save states found in {run_dir}")
      else:
        logging.info(f"Using last saved state, {epochs[0]}")
        state_dict = torch.load(join(run_dir, epochs[0]), map_location="cpu")

  model.load_state_dict(state_dict)
  if device is not None:
    model.to(device)
  model.eval()
  return model
