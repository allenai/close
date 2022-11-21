from typing import Union, Optional, List, Callable, Any, Dict, Tuple
import torch
from allennlp.common import Registrable, FromParams
from allennlp.nn.beam_search import BeamSearch
from dataclasses import dataclass
from torch import nn


BEST_STATE_NAME = "best-state.pth"


@dataclass
class ExampleOutput:
    text: List[str]
    text_logprobs: List[float]

    def set_beams_to_keep(self, n):
        if self.text is None:
            return self
        return ExampleOutput(self.text[:n], self.text_logprobs[:n])


class BeamSearchSpec(FromParams):
    """Specifies how to do beam search"""

    def __init__(self, beam_size, max_seq_len, per_node_beam_size=None, sampler=None):
        self.beam_size = beam_size
        self.per_node_beam_size = per_node_beam_size
        self.sampler = sampler
        self.max_seq_len = max_seq_len

    def build(self, end_index) -> BeamSearch:
        return BeamSearch(
            end_index, self.max_seq_len, self.beam_size,
            self.per_node_beam_size, self.sampler,
        )


class Model(nn.Module, Registrable):
    """Generic model API inherited from GPV 2, basically a pytorch module with additional
    pre-processing and prediction APIs"""

    def initialize(self, load_params=True):
        """Initialize the model by constructing all parameters and buffers needed

        if `load_params` is false, the model should still set up all its parameters and buffers,
        but does not need to fill them with initialized values (e.g., because it will load
        those parameters from state dict).
        """
        raise NotImplementedError()

    def preprocess_example_train(self, example) -> List:
        """Convert a training example for a task into a pre-processed format

        We support a one-to-many mapping for train examples
        """

        # By default, use the general method
        return [self.preprocess_example(example)]

    def preprocess_example(self, example) -> Any:
        """Convert an eval example for a task into a pre-processed format"""
        raise NotImplementedError()

    def get_collate(self, is_train=False) -> Callable[[List], Dict[str, Any]]:
        """Function that maps pre-processed examples to tensors suitable for `forward`"""
        raise NotImplementedError()

    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes the loss and any scalars to log using the outputs of `self.get_collate()(batch)`
        This is used during training.
        """
        raise NotImplementedError()

    def predict(self, *args, **kwargs) -> List:
        """Computes the test-time example outputs for a batch of examples"""
        raise NotImplementedError()

    def set_prediction_args(
        self, *args: Union[str, int, float, BeamSearchSpec],
        **kwargs: Union[str, int, float, BeamSearchSpec]
    ):
        """Sets parameters to be used during prediction"""
        raise NotImplementedError()
