from dataclasses import dataclass
from os.path import isfile, join
from typing import List
import numpy as np

from close import file_paths
from close.data.dataset import Dataset
from close.utils.py_utils import int_to_str, load_json_object


@dataclass
class VisualNewsExample:
  example_id: str
  caption: str
  image_id: str
  article: str

  def get_example_id(self):
    return self.example_id


def _load(file, sample):
  data = load_json_object(file)

  if sample:
    data = sorted(data, key=lambda x: x["id"])
    if sample < 1:
      sample = int(len(data) * sample)
    data = np.random.RandomState(613423).choice(data, sample, replace=False)

  out = []
  for d in data:
    a_path = join(file_paths.VISUAL_NEWS, d["article_path"][2:])
    i_path = join(file_paths.VISUAL_NEWS, d["image_path"][2:])

    if not isfile(a_path) or not isfile(i_path):
      continue

    with open(a_path, 'r') as f:
      article = f.readlines()[0]

    out.append(VisualNewsExample(
      example_id=d["id"],
      caption=d["caption"],
      # NB: The split depends on the image file path. Mine looks like
      # `/home/sophiag/data/visual_news/origin/bbc/images/0013/600.jpg`,
      # and `split("/", 6)[-1]` gives me `bbc/images/0013/600.jpg`.
      image_id=f'visual_news/{i_path.split("/", 6)[-1]}',
      article=article
    ))
  return out


@Dataset.register("visual-news")
class VisualNews(Dataset):

  def __init__(self, split, sample=None):
    if split not in {'train', 'val', 'test'}:
      raise ValueError()
    self.split = split
    self.sample = sample

  def get_name(self) -> str:
    name = f"visual-news-{self.split}"
    if self.sample is not None:
      name += f"-s{int_to_str(self.sample)}"
    return name

  def load(self) -> List[VisualNewsExample]:
    file = join(file_paths.VISUAL_NEWS, f'{self.split}_data.json')
    return _load(file, self.sample)


if __name__ == '__main__':
  print(len(VisualNews("val").load()))
