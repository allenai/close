import argparse
import logging
import os
import sys

root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(root_folder)

from close.data.coco_captioning import CocoCaptioningKP
from close.data.visual_entailment import VisualEntailment
from close.data.vqa_e import EVQA
from close.data.vqa_v2 import Vqa2, VqaWithCaptions
from close.experiments.utils import get_adapter, get_evaluator, get_default_seq_len
from close.train.optimizer import AdamWBuilder, DelayedWarmupScheduleBuilder
from close.train.trainer import TrainerSimple
from close.utils import py_utils

from close.model.clip_t5_model import ClipT5Model, EmbeddingTokenizer
from close.model.model import BeamSearchSpec
from close.train.evaluator import VqaEvaluator, CaptionEvaluator


os.environ["TOKENIZERS_PARALLELISM"] = "false"


DEFAULT_NOISE = {
  "vqa": 0.04,
  "evqa": 0.04,
  "ve": 0.08,
  "s-cap": 0.12,
  "m-cap": 0.04
}


def main():
  parser = argparse.ArgumentParser("Train a CLOSE model")

  parser.add_argument("--data", default="evqa")

  # Model args
  parser.add_argument("--clip_model", default="ViT-L/14")
  parser.add_argument("--t5_model", default="t5-base")
  parser.add_argument("--train_on_images", action="store_true")
  parser.add_argument("--l_adapter", default="noise")
  parser.add_argument("--noise", type=float, default=None)
  parser.add_argument("--scale", type=float)

  # Optimizer args
  parser.add_argument("--lr", type=float, default=3e-4)
  parser.add_argument("--warmup", type=int, default=None)
  parser.add_argument("--decay", default="linear")

  # Other training args
  parser.add_argument("--batch_size", default=None, type=int)
  parser.add_argument("--epochs", default=8, type=int)

  # Where to save things
  parser.add_argument("--override", action="store_true")
  parser.add_argument("--output_dir")

  parser.add_argument("--debug", action="store_true",
                      help="Train with tiny dataset for debugging")

  args = parser.parse_args()
  dbg = args.debug
  py_utils.add_stdout_logger()

  # Figure out the per-dataset settings
  default_batch_size = 128
  caption_mode = "1to1"
  if args.data in {"vqa-e", "vqa", "vqa-trainval"}:
    if args.data == "vqa":
      tr = VqaWithCaptions("train", 50 if dbg else None)
      val = VqaWithCaptions("val", sample=50 if dbg else 5000)
    elif args.data == "vqa-trainval":
      tr = VqaWithCaptions("trainval", 50 if dbg else None)
      val = VqaWithCaptions("val", sample=50 if dbg else 1000)
    else:
      tr = EVQA("train", 50 if dbg else None)
      val = EVQA("val", sample=50 if dbg else 5000)
    default_noise = 0.04
  elif args.data == "ve":
    tr = VisualEntailment("train", sample=8 if dbg else None)
    val = VisualEntailment("val", sample=10 if dbg else None)
    default_noise = 0.08
  elif args.data == "s-cap":
    tr = CocoCaptioningKP("train", 50 if dbg else None)
    val = CocoCaptioningKP("val", 50 if dbg else None)
    default_noise = 0.12
  elif args.data == "m-cap":
    caption_mode = "other-target"
    tr = CocoCaptioningKP("train", 50 if dbg else None)
    val = CocoCaptioningKP("val", 50 if dbg else None)
    default_noise = 0.04
    # Since we will train with about 4 targets per an input caption
    default_batch_size = 32
  else:
    raise NotImplementedError(args.data)

  if args.noise is None:
    logging.info(f"Default to noise {args.noise}")
    args.noise = default_noise

  if args.batch_size is None:
    logging.info(f"Default to batch size {default_batch_size}")
    args.batch_size = default_batch_size

  # Build the model
  l_adapter = get_adapter(args)
  adapter = EmbeddingTokenizer(4)
  if args.clip_model == "open_clip":
    openai_clip = 'laion400m_e32'
    args.clip_model = "ViT-L/14"
  else:
    openai_clip = None

  model = ClipT5Model(
    args.clip_model, args.t5_model, adapter,
    language_shift=l_adapter, lowercase_target=True, train_on_l=not args.train_on_images,
    openai_clip=openai_clip, caption_mode=caption_mode,
    average_vqa_caption=True
  )

  if args.warmup or args.decay:
    scheduler = DelayedWarmupScheduleBuilder(warmup=args.warmup, decay=args.decay)
  else:
    scheduler = None

  # Build the trainer and train
  trainer = TrainerSimple(
    train_dataset=tr,
    optimizer=AdamWBuilder(lr=args.lr),
    epochs=args.epochs,
    eval_dataset=val,
    batch_size=args.batch_size,
    evaluator=get_evaluator(val),
    prediction_args=dict(beam_search_spec=BeamSearchSpec(5, get_default_seq_len(val))),
    scheduler=scheduler,
    save_each_epoch=[args.epochs],
    num_workers=3
  )
  trainer.train(model, args.output_dir, override=args.override)


if __name__ == '__main__':
  main()