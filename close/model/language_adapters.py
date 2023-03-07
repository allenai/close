import pickle
from os.path import join, dirname

import torch
from allennlp.common import Params
from torch.distributions import multivariate_normal

from l2v import file_paths
from l2v.model.layers import Layer
from l2v.utils.py_utils import load_json_object
import numpy as np

from l2v.utils.pytorch_utils import replace_parameters


@Layer.register("normalize")
class Normalize(Layer):
  def forward(self, x):
    return x / x.norm(dim=-1, keepdim=True)


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
    src = join(file_paths.DATA_DIR, "clip-stats")
    with open(join(src, f"{self.src}.pkl"), "rb") as f:
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
    self.register_buffer("shift", torch.as_tensor(shift, dtype=torch.float, device=torch.device("cuda")), persistent=False)

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
    src = join(file_paths.DATA_DIR, "clip-stats")
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
        x = x + dist.sample(x.shape[:1])*3
      else:
        x = x + self._shift
      x = x + torch.randn_like(x)*self.scale
      x = x / x.norm(dim=-1, keepdim=True)
    return x


@Layer.register("linear-adapter")
class LinearAdapter(Layer):
  def __init__(self, src, noise, renorm=False):
    super().__init__()
    self.renorm = renorm
    self.src = src
    self.noise = noise
    src = join(file_paths.DATA_DIR, "adapters")
    with open(join(src, f"{self.src}.pkl"), "rb") as f:
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


@Layer.register("coco-cap-mean-diff")
class CocoCapMeanDiff(Layer):
  def __init__(self):
    super().__init__()
    with open(file_paths.COCO_CAP_MEAN_DIFF, "rb") as f:
      data = pickle.load(f)
    self.register_buffer("shift", torch.as_tensor(data), persistent=False)

  def forward(self, x):
    return x - self.shift


@Layer.register("trained-adapter")
class TrainedAdapter(Layer):
  def __init__(self, src, noise=None):
    super().__init__()
    self.src = src
    self.noise = noise
    self.adapter = load_adapter(src)
    replace_parameters(self.adapter, False)

  def forward(self, x):
    out = self.adapter(x)
    if self.noise:
      out = torch.randn_like(out)*self.gnoise
    return out


def load_adapter(src):
  state = torch.load(join(src, "best-state.pth"))
  model = load_json_object(join(dirname(src), "model.json"))
  model = Layer.from_params(Params(model["aligner"]))
  state = {k.split(".", 1)[1]: v for k, v in state.items() if k.startswith("aligner.")}
  model.load_state_dict(state)
  return model
