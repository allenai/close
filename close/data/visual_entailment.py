import json
from dataclasses import dataclass
from os.path import join
from typing import Dict, List, Any, Optional

from close import file_paths
from close.data.dataset import Dataset
from close.utils import py_utils
from close.utils.py_utils import int_to_str


@dataclass
class VisualEntailmentExample:
  example_id: str
  image_id: Optional[str]
  label: str
  hypothesis: str
  premise: str


@Dataset.register("snli-ve")
class VisualEntailment(Dataset):

  def __init__(self, split, sample=None, use_images=True):
    self.split = split
    self.sample = sample
    self.use_images = use_images

  def get_name(self) -> str:
    if self.use_images:
      text = "snli"
    else:
      text = "snli-ve"
    text += f"-{self.split}"
    if self.sample is not None:
      text += f"-s{int_to_str(self.sample)}"
    return text

  def load(self):
    out = []
    split = self.split
    if split == "val":
      split = "dev"
    src = join(file_paths.SNLI_VE_HOME, f"snli_ve_{split}.jsonl")
    with open(src) as f:
      lines = f.readlines()

    lines = py_utils.subsample(lines, self.sample, 132124)

    for line in lines:
      example = json.loads(line)
      image_id = "flicker30k/" + example["Flickr30K_ID"] + ".jpg"
      out.append(VisualEntailmentExample(
        example_id="snli-ve/" + example["pairID"],
        image_id=image_id if self.use_images else None,
        label=example["gold_label"],
        hypothesis=example["sentence2"],
        premise=example["sentence1"]
      ))
    return out

