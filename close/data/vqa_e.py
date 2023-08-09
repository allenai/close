import json
import logging
from collections import Counter
from os.path import join
from typing import List

from close import file_paths
from close.data.dataset import Dataset
from close.data.vqa_v2 import VqaExample
from close.utils import py_utils
from close.utils.image_utils import get_coco_image_id
from close.utils.py_utils import int_to_str


@Dataset.register("evqa")
class EVQA(Dataset):

  def __init__(self, split, sample=None, load_answer_types=True):
    self.sample = sample
    self.split = split
    self.load_answer_types = load_answer_types

  def get_name(self) -> str:
    name = f"evqa-{self.split}"
    if self.sample is not None:
      name += f"-s{int_to_str(self.sample)}"
    return name

  def load(self) -> List:
    e_file = join(file_paths.VQAE, f"VQA-E_{self.split}_set.json")
    logging.info(f"Loading {e_file}")
    with open(e_file) as f:
      data = json.load(f)

    answer_types = {}
    if self.load_answer_types:
      with open(join(file_paths.VQA_ANNOTATIONS, f"v2_mscoco_{self.split}2014_annotations.json")) as f:
        annotations = json.load(f)
      for anno in annotations["annotations"]:
        key = tuple(sorted(x["answer"] for x in anno["answers"]))
        assert key not in annotations
        answer_types[key] = anno["answer_type"]

    out = []
    for ix, ex in enumerate(data):
      out.append(VqaExample(
        example_id=f'image{ex["img_id"]}-id{ix}',
        question=ex["question"],
        answers=Counter(ex["answers"]),
        multiple_choice_answer=ex["multiple_choice_answer"],
        question_type=ex["question_type"],
        answer_type=answer_types[tuple(sorted(ex["answers"]))],
        image_text=ex["explanation"][0].strip(),
        image_id=get_coco_image_id(self.split + "2014", ex["img_id"])
      ))
    return py_utils.subsample(out, self.sample, 16914)
