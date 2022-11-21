import logging
from typing import Union

import torch
from torch import nn


def get_device(device_name: Union[None, str, int]=None):
  if device_name is None:
    if torch.cuda.is_available():
      logging.info("cuda found, defaulting to cuda device")
      return torch.device('cuda')
    else:
      logging.info("cuda not found, using cpu device")
      return torch.device('cpu')
  else:
    try:
      device_name = int(device_name)
    except ValueError:
      pass
    return torch.device(device_name)


def to_device(batch, device):
  if batch is None:
    return None
  if isinstance(batch, (float, int, str)):
    return batch
  if isinstance(batch, dict):
    return {sub_k: to_device(sub_v, device) for sub_k, sub_v in batch.items()}
  if isinstance(batch, (tuple, list)):
    return [to_device(x, device) for x in batch]
  else:
    return batch.to(device)


def get_model_device(module: torch.nn.Module):
  return next(module.parameters()).device


def replace_parameters(model: nn.Module, persistent):
  """Replace's the model parameters with buffers"""
  for child in model.modules():
    for name, param in list(child.named_parameters(recurse=False)):
      child.__delattr__(name)
      child.register_buffer(name, param.data, persistent)


def segment_mean(x, segments):
  counts = torch.unique_consecutive(segments.cpu(), return_counts=True)[1]
  start = 0
  means = []
  for c in counts:
    means.append(x[start:start+c].mean(0))
    start += c
  return torch.stack(means, 0)


def concat_masked_sequences(
    seq1, seq2, mask2=None
):
  batch = seq1.size(0)
  if mask2 is None:
    return torch.cat([seq1, seq2], 1), None
  else:
    out = torch.cat([seq1, seq2], 1)
    mask = torch.cat([
      torch.ones(batch, seq1.size(1), device=seq1.device, dtype=mask2.dtype),
      mask2
    ], 1)
    return out, mask
