import pickle
from os.path import join, dirname

import torch
from allennlp.common import Params
from torch.distributions import multivariate_normal

from close import file_paths
from close.model.layers import Layer
from close.utils.py_utils import load_json_object
import numpy as np

from close.utils.pytorch_utils import replace_parameters


@Layer.register("add-guassian-noise")
class AddGuassianNoise(Layer):
  def __init__(self, scale: float, renormalize=True):
    super().__init__()
    self.scale = scale
    self.renormalize = renormalize

  def forward(self, x):
    if self.training:
      x = x + torch.randn_like(x)*self.scale
      if self.renormalize:
        x = x / x.norm(dim=-1, keepdim=True)
    return x


@Layer.register("mean-shift")
class Shift(Layer):
  def __init__(self, src, scale, noise, renorm=False):
    super().__init__()
    self.renorm = renorm
    self.src = src
    self.noise = noise
    self.scale = scale
    with open(join(file_paths.ADAPTER_SOURCE, f"{self.src}.pkl"), "rb") as f:
      shift = pickle.load(f)
    self.register_buffer("_shift", torch.as_tensor(shift*scale, dtype=torch.float), persistent=False)

  def forward(self, x):
    x = x + self._shift
    if self.renorm:
      x = x / x.norm(dim=-1, keepdim=True)
    if self.training:
      x = x + torch.randn_like(x)*self.noise
      x = x / x.norm(dim=-1, keepdim=True)
    return x


@Layer.register("random-shift")
@Layer.register("fixed-random-shift")
class FixedRandomShift(Layer):

  def __init__(self, scale, noise, dim, seed=None, renorm=False):
    super().__init__()
    self.renorm = renorm
    self.noise = noise
    self.seed = seed
    self.scale = scale
    self.dim = dim
    shift = np.random.randn(dim)
    shift = scale * shift / np.linalg.norm(shift)
    self.register_buffer("shift", torch.as_tensor(
      shift, dtype=torch.float, device=torch.device("cuda")), persistent=False)

  def forward(self, x):
    x = x + self.shift
    if self.renorm:
      x = x / x.norm(dim=-1, keepdim=True)
    if self.training and self.noise:
      x = x + torch.randn_like(x)*self.noise
      x = x / x.norm(dim=-1, keepdim=True)
    return x


@Layer.register("cov-noise")
class CovNoise(Layer):
  def __init__(self, src, scale, shift=True, cov=True, version=2):
    super().__init__()
    self.version = version
    self.src = src
    self.cov = cov
    self.shift = shift
    self.scale = scale
    src = file_paths.ADAPTER_SOURCE
    if self.shift:
      with open(join(src, f"{self.src}-mean-diff.pkl"), "rb") as f:
        shift = pickle.load(f)
      self.register_buffer("_shift", torch.as_tensor(shift, dtype=torch.float), persistent=False)
    if self.cov:
      with open(join(src, f"{self.src}-cov-diff.pkl"), "rb") as f:
        cov = pickle.load(f)
      self.register_buffer("_cov", torch.as_tensor(cov, dtype=torch.float), persistent=False)

  def forward(self, x):
    if self.training:
      device = x.device
      if self.cov:
        if self.shift:
          x = x + torch.unsqueeze(self._shift, 0)
        dist = multivariate_normal.MultivariateNormal(
          loc=torch.zeros(x.shape[1], device=device), covariance_matrix=self._cov)
        x = x + dist.sample(x.shape[:1])*self.scale
      else:
        x = x + self._shift
        x = x + torch.randn_like(x)*self.scale
      x = x / x.norm(dim=-1, keepdim=True)
    elif self.shift:
      x = x + self._shift
      x = x / x.norm(dim=-1, keepdim=True)
    return x


@Layer.register("linear-adapter")
class LinearAdapter(Layer):
  def __init__(self, src, noise, renorm=False):
    super().__init__()
    self.renorm = renorm
    self.src = src
    self.noise = noise
    with open(join(file_paths.ADAPTER_SOURCE, f"{self.src}.pkl"), "rb") as f:
      coef, bias = pickle.load(f)
    self.register_buffer("coef", torch.as_tensor(coef, dtype=torch.float), persistent=False)
    self.register_buffer("bias", torch.as_tensor(bias, dtype=torch.float), persistent=False)

  def forward(self, x):
    x = torch.matmul(x, self.coef.T) + torch.unsqueeze(self.bias, 0)
    if self.renorm:
      x = x / x.norm(dim=-1, keepdim=True)
    if self.training:
      x = x + torch.randn_like(x)*self.noise
      x = x / x.norm(dim=-1, keepdim=True)
    return x
