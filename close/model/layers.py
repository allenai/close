from os.path import join, dirname
from typing import List, Dict, Any

import torch
from allennlp.common import Registrable, FromParams
from torch import nn

from close.utils import pytorch_utils
from close.utils.py_utils import load_json_object
from close.utils.to_params import to_params


class Layer(nn.Module, Registrable):
  pass


@Layer.register("seq")
class Sequential(Layer, nn.Sequential):
  def __init__(self, args: List[Layer]):
    super(Sequential, self).__init__(*args)

  def _to_params(self) -> Dict[str, Any]:
    return dict(args=[to_params(x, Layer) for x in self])


@Layer.register("normalize")
class Normalize(Layer):
  def forward(self, x):
    return x / x.norm(dim=-1, keepdim=True)

