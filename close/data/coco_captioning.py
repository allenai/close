import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from os.path import join
from typing import Optional, Dict, Any, List

from close import file_paths
from close.data.dataset import Dataset
from close.utils import image_utils, py_utils
from close.utils.py_utils import int_to_str


@dataclass(frozen=True)
class CaptioningExample:
  example_id: str
  image_id: Optional[str]
  captions: List[str]
  meta: Optional[Dict[str, Any]] = None


@Dataset.register("coco-kp-cap")
class CocoCaptioningKP(Dataset):

  def __init__(self, split, sample=None):
    self.split = split
    self.sample = sample

  def get_name(self) -> str:
    name = f"coco-cap-kp-{self.split}"
    if self.sample is not None:
      name += f"-s{int_to_str(self.sample)}"
    return name

  def load(self) -> List:
    with open(join(file_paths.COCO_SOURCE, "dataset_coco.json")) as f:
      data = json.load(f)["images"]
    examples = [x for x in data if x["split"] == self.split]
    out = []
    for ex in examples:
      image_id = join("coco", ex["filepath"], ex["filename"])
      out.append(CaptioningExample(
        f"coco-cap-{ex['imgid']}'", image_id,
        [x["raw"] for x in ex["sentences"]],
      ))
    return py_utils.subsample(out, self.sample, 96810)


ANNOTATION_FILE_MAP = {
  "train": "captions_train2014.json",
  "val": "captions_val2014.json",
  "test": "captions_test2014.json"
}


def _load(caption_file, sample=None):
  logging.info(f"Loading captioning data from {caption_file}")
  with open(caption_file) as f:
    data = json.load(f)

  subset = caption_file.split("_")[1].split(".")[0]
  assert subset in {"train2014", "val2014"}

  image_id_to_cap = defaultdict(list)
  for anno in data["annotations"]:
    image_id_to_cap[anno["image_id"]].append(anno)

  image_ids = image_id_to_cap
  if sample:
    image_ids = py_utils.subsample(image_ids, sample, 613423, lambda x: x)

  out = []
  for image_id in image_ids:
    caps = image_id_to_cap[image_id]
    cap_objects = []
    for cap in caps:
      cap_objects.append(cap['caption'])

    out.append(CaptioningExample(
      f"coco-cap-{image_id}",
      image_utils.get_coco_image_id(subset, image_id),
      cap_objects
    ))

  return out


@Dataset.register("coco-cap")
class CocoCaptioning2014(Dataset):

  def __init__(self, split, sample=None):
    if split not in ANNOTATION_FILE_MAP:
      raise ValueError()
    self.split = split
    self.sample = sample

  def get_name(self) -> str:
    name = f"coco-cap-{self.split}"
    if self.sample is not None:
      name += f"-s{int_to_str(self.sample)}"
    return name

  def load(self) -> List[CaptioningExample]:
    return _load(join(file_paths.COCO_ANNOTATIONS, ANNOTATION_FILE_MAP[self.split]), self.sample)

