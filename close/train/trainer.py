import json
import logging
import os
import socket
from datetime import datetime
from os import makedirs
from os.path import join, exists
from time import perf_counter
from typing import List, Optional, Dict, Union

import numpy as np
import torch
from allennlp.common import FromParams, Params
from dataclasses import dataclass
from random import shuffle
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from close.data.dataset import Dataset
from close.model.model import Model, BEST_STATE_NAME, BeamSearchSpec
from close.train.evaluator import Evaluator, ResultKey
from close.train.optimizer import OptimizerBuilder, TrainingScheduleBuilder
from close.train.runner import CollateWithBatch
from close.utils import py_utils, pytorch_utils
from close.utils.py_utils import dump_json_object
from close.utils.to_params import to_params


def select_subdir(output_dir, target=None):
  prefix = "" if target is None else target + "-"
  i = 0
  while True:
    candidate = join(output_dir, prefix + "r" + str(i))
    if not exists(candidate):
      try:
        os.mkdir(candidate)
        return candidate
      except FileExistsError:
        pass
    i += 1


@dataclass
class TrainerSimple(FromParams):
  """Class to run the training loop for our models"""

  train_dataset: Dataset
  """Datast to train on"""

  optimizer: OptimizerBuilder
  """Optimizer to use"""

  epochs: int
  """Number of epochs to train on"""

  batch_size: int
  """Batch size to train with"""

  # Evaluation parameter
  eval_dataset: Dataset = None
  """Evaluation dataset to evaluate on every epoch"""

  evaluator: Evaluator = None
  """Evaluator to use for evaluations"""

  prediction_args: Dict[str, Union[int, float, str, BeamSearchSpec]]=None
  """Test-time args (e.g., beam size) to use during evaluation"""

  # Dataloader parameters
  num_workers: int = 4
  pin_memory: bool = True

  # Additional optimization settings
  scheduler: TrainingScheduleBuilder = None
  """Supports a learning rate shedule to adjust the learning rate each step"""

  clip_grad_norm: Optional[float] = None
  """Do gradient norm clipping"""

  # Saving the results
  save_evaluation_results: bool = True
  """Should we save the evaluation results in json files"""

  save_each_epoch: Union[int, List[int]] = True
  """Should we save the model each epoch"""

  best_model_key: ResultKey = None
  """Keep track of the best model weights using this metrics from `self.evaluator`"""

  # Cosmetic/Logging
  tb_log: bool = True
  """Should we log to tensorboard"""

  tb_log_intervals: int = 20
  """How often to log per-train-step metrics to tensorboard"""

  log_lr: bool = True
  """Should the learning rate be logged to tensorboard"""

  loss_logging_ema: float = 0.99
  """Decay factor for exponential moving average of the loss"""

  monitor_ema: float = 0.99
  """Decay factor for exponential moving average of other metrics"""

  eval_pbar: bool = True
  """Show a progress bar when evaluating"""

  epoch_pbar: bool = True
  """Show a progress bar when training"""

  def train(self, model: Model, output_dir: Optional[str],
            device: Optional[int]=None, override=False):
    """Train a model

    :param model: Model to train
    :param output_dir: directory to save reults
    :param device: GPU device to train on
    :param override: Override `output_dir` if it exists
    """
    if output_dir is not None:
      logging.info(f"Initializing model dir {output_dir}")
      py_utils.clear_if_nonempty(output_dir, override)
      makedirs(output_dir, exist_ok=True)
      Params(to_params(self)).to_file(join(output_dir, "trainer.json"))
      Params(to_params(model, Model)).to_file(join(output_dir, "model.json"))
    else:
      logging.info(f"No output dir, model will not be saved")
    return self._train(model, output_dir, device)

  @staticmethod
  def train_another_model(output_dir: str, device: Optional[int]=None, save=True):
    """Train another run of the model stored in `output_dir`

    :param output_dir:
    :param device: Devices to train on
    :param save: Save the new run in `output_dir`, otherwise do not save anything
    """
    logging.info(f"Starting another run for {output_dir}")
    logging.info("Getting trainer/model")
    with py_utils.DisableLogging():
      trainer = TrainerSimple.from_params(Params.from_file(join(output_dir, "trainer.json")))
      model = Model.from_params(Params.from_file(join(output_dir, "model.json")))

    if not save:
      logging.info("Save is false, so no results will be recorded")
      output_dir = None
    return trainer._train(model, output_dir, device)

  def _train(self, model: Union[str, Model], output_dir, device):
    """Train with the output dir already initialized, and possibly a checkpoint file"""
    if device is None:
      device = pytorch_utils.get_device()

    if output_dir is not None:
      # Build the dir to save our configuration, log the metrics, and save the model state
      logging.info("Initializing run dir")
      run_dir = select_subdir(output_dir)
      with open(join(run_dir, "runtime.json"), "w") as f:
        json.dump(dict(
          hostname=socket.gethostname(),
          date=datetime.now().strftime("%m%d-%H%M%S"),
          device=str(device)
        ), f, indent=2)
      dump_json_object(dict(done=False), join(run_dir, "status.json"))

      log_file = join(run_dir, "out.log")
      record_log_handle = logging.FileHandler(log_file)
      logging.getLogger().addHandler(record_log_handle)
    else:
      # Not saving anything to disk
      run_dir = None

    MAX_TRAIN_EXAMPLES = 200_000
    MAX_EVAL_EXAMPLES = 20_000

    device = pytorch_utils.get_device()
    logging.info(f"Initializing model on {device}")
    model.initialize()
    model.to(device)
    if self.prediction_args is not None:
      model.set_prediction_args(**self.prediction_args)

    logging.info("Loading training data")
    training_examples = self.train_dataset.load()
    training_examples = py_utils.flatten_list(model.preprocess_example_train(x) for x in training_examples)

    if self.eval_dataset is not None:
      assert self.evaluator is not None
      logging.info("Loading eval data")
      eval_examples = self.eval_dataset.load()

    logging.info("Preparing optimizers")
    optimizer = self.optimizer.build(model, min(len(training_examples), MAX_TRAIN_EXAMPLES), self.epochs)
    if self.scheduler is not None:
      schedule = self.scheduler.build(optimizer, min(len(training_examples), MAX_TRAIN_EXAMPLES)*self.epochs, 0)
    else:
      schedule = None

    # Other stuff we need to track during training for logging metrics
    if run_dir and self.tb_log:
      summary_writer = SummaryWriter(join(run_dir, "log"))
    else:
      summary_writer = None
    best_saved_score = None
    monitor_ema = {}
    loss_ema = 0
    global_step = 0

    n_train = sum(p.requires_grad for p in model.parameters())
    n_freeze = sum(not p.requires_grad for p in model.parameters())
    logging.info(f"Have {n_train} params and {n_freeze} frozen parameters")
    logging.info(f"Start training")

    for epoch in range(0, self.epochs):
      ep_start = perf_counter()
      model.train()

      # tqdm to get a progress bar for the DataLoader
      shuffle(training_examples)
      train_loader = DataLoader(
        training_examples[:MAX_TRAIN_EXAMPLES], self.batch_size,
        shuffle=True, num_workers=self.num_workers,
        collate_fn=model.get_collate(True),
        pin_memory=self.pin_memory,
      )
      pbar = tqdm(train_loader, disable=not self.epoch_pbar, ncols=100,
                  desc="loss=", total=len(train_loader))

      for batch in pbar:
        batch = pytorch_utils.to_device(batch, device)
        loss, monitor = model(**batch)
        loss.backward()
        loss = loss.item()

        if self.clip_grad_norm is not None:
          torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_grad_norm)

        optimizer.step()
        if not np.isfinite(loss):
          raise ValueError(f"non-finite foss {loss}")

        # Manually remove gradients, slightly faster then `optimizer.zero_grad`
        for group in optimizer.param_groups:
          for p in group['params']:
            p.grad = None

        global_step += 1
        if schedule is not None:
          # Have to call step manually
          schedule.step()

        for metric_name, metric_value in monitor.items():
          # Compute the exponential moving average for the metrics
          if metric_name not in monitor_ema:
            monitor_ema[metric_name] = metric_value
            to_show = metric_value
          else:
            cur = monitor_ema[metric_name]
            ema = cur * self.monitor_ema + metric_value * (1 - self.monitor_ema)
            monitor_ema[metric_name] = ema
            to_show = (ema / (1 - self.monitor_ema ** global_step))

          # Write to tensorboard
          if summary_writer is not None and global_step % self.tb_log_intervals == 0:
            summary_writer.add_scalar(f"train/{metric_name}", to_show, global_step)

        # Compute the exponential moving average of the loss
        loss_ema = loss_ema * self.loss_logging_ema + loss * (1 - self.loss_logging_ema)
        corrected_loss_ema = (loss_ema / (1 - self.loss_logging_ema ** global_step))

        # Set it to the pbar description
        pbar.set_description("loss=%.4f" % corrected_loss_ema, refresh=False)

        if summary_writer is not None and global_step % self.tb_log_intervals == 0:
          # Write the loss to tensorboard
          summary_writer.add_scalar("train/loss-smoothed", corrected_loss_ema, global_step)
          summary_writer.add_scalar("train/loss", loss, global_step)

          if self.log_lr:
            # Write the loss to tensorboard, useful to check what the learnign shedule is doing
            for j, group in enumerate(optimizer.param_groups):
              name = group.get("name", f"group_{j}")
              summary_writer.add_scalar(f'lr/{name}', group["lr"], global_step)
        break

      ep_end = perf_counter()
      logging.info(f"Epoch {epoch + 1} took {py_utils.duration_to_str(ep_end - ep_start)}, starting evaluation")

      eval_start = perf_counter()
      model.eval()
      predictions = {}
      shuffle(eval_examples)
      eval_loader = DataLoader(
        [model.preprocess_example(x) for x in eval_examples[:MAX_EVAL_EXAMPLES]],
        batch_size=self.batch_size,
        collate_fn=CollateWithBatch(model.get_collate()),
        num_workers=self.num_workers,
        shuffle=False,
        pin_memory=self.pin_memory
      )
      it = tqdm(eval_loader, desc="eval", ncols=100, disable=not self.eval_pbar)
      for examples, batch in it:
        batch = pytorch_utils.to_device(batch, device)
        with torch.no_grad():
          output = model.predict(**batch)
        for ex, out in zip(examples, output):
          predictions[ex.get_example_id()] = out
      results = self.evaluator.evaluate(eval_examples[:MAX_EVAL_EXAMPLES], predictions)
      eval_end = perf_counter()
      logging.info(f"Evaluation {epoch + 1} took {py_utils.duration_to_str(eval_end - eval_start)}")

      for k, v in results.items():
        if isinstance(v, float):
          v = "%0.4f" % v
        logging.info(f"{k}={v}")

      if summary_writer:
        # Log evaluation result o tensorboard
        summary_writer.add_scalar("time/train", ep_end-ep_start, epoch+1)
        summary_writer.add_scalar("time/eval", eval_end - eval_start, epoch + 1)
        for key, val in results.items():
          summary_writer.add_scalar(str(key), val, global_step)

      if self.best_model_key:
        # Check is this is the best model so far according to `self.best_model_key`,
        # if so save as our best set of weights
        score = results[self.best_model_key]
        if best_saved_score is None or best_saved_score < score:
          prefix = "Saving as" if run_dir else "Found"
          if best_saved_score is None:
            logging.info(f"{prefix} best model ({score:.5f}) ep={epoch+1}")
          else:
            logging.info(f"{prefix} best model ({score:.5f} > {best_saved_score:.5f}) ep={epoch+1}")
          best_saved_score = score
          if run_dir:
            best_model_file = join(run_dir, BEST_STATE_NAME)
            torch.save(model.state_dict(), best_model_file)

      if run_dir is not None:
        if self.save_each_epoch and (
            (isinstance(self.save_each_epoch, list) and (epoch+1) in self.save_each_epoch) or
            (isinstance(self.save_each_epoch, int) and (epoch+1) % self.save_each_epoch == 0)
        ):
          state_file = join(run_dir, f"state-ep{epoch+1}.pth")
          logging.info(f"Saving state as {state_file}")
          torch.save(model.state_dict(), state_file)
