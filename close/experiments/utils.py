import os
from typing import Union

from close.data.coco_captioning import CocoCaptioningKP
from close.data.dataset import Dataset
from close.data.vqa_e import EVQA
from close.data.vqa_v2 import Vqa2, VqaWithCaptions
from close.data.visual_entailment import VisualEntailment
from close.model.language_adapters import *
from close.train.evaluator import CaptionEvaluator, Evaluator, VqaEvaluator, EntailmentEvaluator

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_default_seq_len(ds: Dataset) -> int:
  if isinstance(ds, (CocoCaptioningKP,)):
    return 32
  if isinstance(ds, (Vqa2, EVQA, VqaWithCaptions)):
    return 16
  if isinstance(ds, (VisualEntailment,)):
    return 8
  else:
    raise NotImplementedError(f"No default lengths set for dataset {ds}")


def get_evaluator(ds: Dataset) -> Union[Evaluator, None]:
  if isinstance(ds, (CocoCaptioningKP,)):
    return CaptionEvaluator()
  if isinstance(ds, (Vqa2, EVQA, VqaWithCaptions)):
    return VqaEvaluator()
  if isinstance(ds, (VisualEntailment,)):
    return EntailmentEvaluator()
  else:
    raise ValueError()


def get_adapter(args):
  if args.l_adapter.startswith("shift"):
    _, src = args.l_adapter.split("-", maxsplit=1)
    return Shift(src, args.scale, args.noise, renorm=False)
  elif args.l_adapter == "lin":
    return LinearAdapter("kp-linear-v1", args.noise, renorm=True)
  elif args.l_adapter == "cc3m-lin":
    return LinearAdapter("cc3m-linear-v1", args.noise, renorm=True)
  elif args.l_adapter == "cov":
    return CovNoise("kp-cov-v1", args.noise)
  elif args.l_adapter == "cc3m-cov":
    return CovNoise("cc3m-cov-v1", args.noise)
  elif args.l_adapter == "noise":
    return AddGuassianNoise(args.noise, renormalize=True)
  elif args.l_adapter == "vis-news-shift":
    l_adapter = CovNoise("kp-restval", args.noise, cov=False)
  elif args.l_adapter == "vis-news-cov":
    l_adapter = CovNoise("kp-restval", args.noise)
  elif args.l_adapter is None:
    raise NotImplementedError()
