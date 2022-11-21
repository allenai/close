import torch
from typing import Dict, Tuple, List, Optional, Any, Union

from allennlp.common import Registrable
from dataclasses import dataclass
from torch.optim import AdamW, SGD, Optimizer


class OptimizerBuilder(Registrable):
  """Builds an Optimizer

  We use this class rather then using an Optimizer directly since it can be
  serialized with FromParams, and can dynamically determine how to handle groups
  of parameters depending on the model
  """

  def build(self, model, epoch_size, n_epochs) -> Optimizer:
    raise NotImplementedError()


class TrainingScheduleBuilder(Registrable):
  """Builds an learning rate schedule"""

  def build(self, optimizer, num_steps, last_epoch):
    raise NotImplementedError()


def _per_or_int_to_int(x, total):
  if isinstance(x, int):
    return x
  return round(x*total)


class DelayedWarmupSchedule:

  def __init__(
      self,
      optimizer,
      warmup: Union[int, float],
      total: int,
      decay="linear",
  ):
    self.warmup = 0 if warmup is None else warmup
    self.total = total
    self.optimizer = optimizer
    self.decay = decay
    self._step = 0
    for group in optimizer.param_groups:
      group["initial_lr"] = group["lr"]

  def state_dict(self):
    return dict(step=self._step)

  def load_state_dict(self, state):
    self._step = state["step"]

  def step(self):
    self._step += 1
    for group in self.optimizer.param_groups:
      wu = group.get("warmup", self.warmup)
      wu = _per_or_int_to_int(wu, self.total)
      if self._step < wu:
        factor = self._step / wu
      else:
        decay = group.get("decay", self.decay)
        if decay == "linear":
          factor = (self.total - self._step) / (self.total - wu)
        elif decay is None or decay == "none":
          factor = 1.0
        else:
          raise NotImplementedError()
      group["lr"] = group["initial_lr"] * factor


@dataclass
@TrainingScheduleBuilder.register("delayed-warmup-linear")
class DelayedWarmupScheduleBuilder(TrainingScheduleBuilder):
  warmup: Union[int, float, None] = 0
  decay: Optional[str] = "linear"

  def build(self, optimizer, num_steps, last_epoch):
    return DelayedWarmupSchedule(optimizer, self.warmup, num_steps, self.decay)


@OptimizerBuilder.register("adam-w")
@dataclass
class AdamWBuilder(OptimizerBuilder):
  lr: float
  weight_decay: float = 0.0
  betas: Tuple[float, float] = (0.9, 0.999)

  def build(self, model: torch.nn.Module, epoch_size, n_epochs):
    return AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay, betas=self.betas)

