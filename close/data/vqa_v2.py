from dataclasses import dataclass, replace
from os.path import join
from typing import List, Optional, Union
import numpy as np

from collections import Counter

from close import file_paths
from close.data.coco_captioning import CocoCaptioning2014
from close.data.dataset import Dataset
from close.utils import image_utils, py_utils
from close.utils.py_utils import int_to_str, load_json_object

ANNOTATION_FILE_MAP = {
  "train": [
    "v2_OpenEnded_mscoco_train2014_questions.json",
    "v2_mscoco_train2014_annotations.json"
  ],
  "val": [
    "v2_OpenEnded_mscoco_val2014_questions.json",
    "v2_mscoco_val2014_annotations.json"
  ],
  "test": [
    "v2_OpenEnded_mscoco_test2015_questions.json",
    None
  ]
}


@dataclass
class VqaExample:
    example_id: str
    question: str
    image_id: str
    question_type: str
    answers: Counter
    image_text: Union[None, str, List[str]] = None
    multiple_choice_answer: str = None
    answer_type: Optional[str] = None
    
    def get_example_id(self):
        return self.example_id


def _load(q_file, a_file, sample, subset) -> List[VqaExample]:
  q_data = load_json_object(q_file)["questions"]

  if sample:
    q_data = sorted(q_data, key=lambda x: x["question_id"])
    q_data = np.random.RandomState(613423).choice(q_data, sample, replace=False)

  if a_file:
    a_data = load_json_object(a_file)
    anno_map = {}
    for anno in a_data["annotations"]:
      anno_map[anno["question_id"]] = anno
  else:
    anno_map = None

  out = []
  for q in q_data:
    anno = None if anno_map is None else anno_map[q["question_id"]]
    image_id = image_utils.get_coco_image_id(subset, q["image_id"])
    out.append(VqaExample(
      q["question_id"], q["question"], image_id,
      question_type=None if anno is None else anno["question_type"],
      answers=None if anno is None else Counter(x["answer"] for x in anno["answers"]),
      image_text=None
    ))
  return out


@Dataset.register("vqa-v2")
class Vqa2(Dataset):

  def __init__(self, split, sample=None):
    if split not in ANNOTATION_FILE_MAP:
      raise ValueError()
    self.split = split
    self.sample = sample

  def get_name(self) -> str:
    name = f"vqa-{self.split}"
    if self.sample is not None:
      name += f"-s{int_to_str(self.sample)}"
    return name

  def load(self) -> List[VqaExample]:
    q_file, a_file = ANNOTATION_FILE_MAP[self.split]
    q_file = join(file_paths.VQA_ANNOTATIONS, q_file)
    a_file = None if a_file is None else join(file_paths.VQA_ANNOTATIONS, a_file)
    if self.split == "test":
      subset = "test2015"
    else:
      subset = self.split + "2014"
    return _load(q_file, a_file, self.sample, subset)


@Dataset.register("paired-vqa")
class VqaWithCaptions(Dataset):

  def __init__(self, split, sample=None):
    self.split = split
    self.sample = sample

  def get_name(self) -> str:
    return "vqa2-cap"

  def load(self) -> List:
    if self.split == "trainval":
      splits = ["train", "val"]
    else:
      splits = [self.split]

    out = []
    for split in splits:
      vqa = Vqa2(split).load()
      cap = CocoCaptioning2014(split).load()
      image_id_to_cap = {x.image_id: x for x in cap}
      for ex in vqa:
        cap = image_id_to_cap[ex.image_id]
        out.append(replace(ex, image_text=cap.captions))
    return py_utils.subsample(out, self.sample, 281741)
