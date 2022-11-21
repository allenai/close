from typing import List

from allennlp.common import Registrable


class Dataset(Registrable):
  """Dataset we can train/evaluate on"""

  def get_name(self) -> str:
    """Get the name of the dataset that uniquely identifies it"""
    raise NotImplementedError()

  def load(self) -> List:
    """Loads the examples"""
    raise NotImplementedError()
