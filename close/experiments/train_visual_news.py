import argparse
import logging
import os

from transformers import AutoConfig

from l2v.data.visual_news import VisualNews
from l2v.experiments.utils import get_adapter
from l2v.train.optimizer import AdamWBuilder, DelayedWarmupScheduleBuilder
from l2v.train.trainer import TrainerSimple
from l2v.utils import py_utils


os.environ["TOKENIZERS_PARALLELISM"] = "false"

from l2v.model.clip_t5_model import ClipT5Model, LinearAdapter, CLIP_DIMS
from l2v.model.model import BeamSearchSpec
from l2v.train.evaluator import ResultKey, VisualNewsEvaluator
from l2v.utils.pytorch_utils import get_devices


def main():
  parser = argparse.ArgumentParser()

  # Model args
  parser.add_argument("--clip_model", default="open_clip")
  parser.add_argument("--t5_model", default="t5-base")

  parser.add_argument("--train_l", default="always") # change to "both" to also train on images
  parser.add_argument("--cap_l", default="1to1")
  parser.add_argument("--l_adapter", default="none")
  parser.add_argument("--noise", type=float, default=0.0)
  parser.add_argument("--scale", type=float)

  # Optimizer args
  parser.add_argument("--lr", type=float, default=3e-4)
  parser.add_argument("--warmup", type=int, default=None)
  parser.add_argument("--decay", default="linear")

  # Other training args
  parser.add_argument("--batch_size", default=16, type=int)
  parser.add_argument("--epochs", default=8, type=int)

  # Where to save things
  parser.add_argument("--override", action="store_true")
  parser.add_argument("--output_dir")

  parser.add_argument("--debug", action="store_true",
                      help="Train with tiny model/dataset for debugging")

  args = parser.parse_args()

  py_utils.add_stdout_logger()
  l_adapter = get_adapter(args)

  adapter = LinearAdapter(4)

  if args.cap_l is not None and args.train_l is None:
    args.train_l = "always"

  if args.clip_model == "open_clip":
    openai_clip = 'laion400m_e32'
    args.clip_model = "ViT-L/14"
  else:
    openai_clip = None

  dbg = args.debug
  model = ClipT5Model(args.clip_model, args.t5_model, adapter,
                      caption_l=args.cap_l, language_shift=l_adapter,
                      lowercase_target=True, train_on_l=args.train_l,
                      one_to_many_loss="sum", openai_clip=openai_clip)

  if args.warmup or args.decay:
    scheduler = DelayedWarmupScheduleBuilder(warmup=args.warmup, decay=args.decay)
  else:
    scheduler = None

  trainer = TrainerSimple(
    train_dataset=VisualNews("train", sample=32 if dbg else None),
    optimizer=AdamWBuilder(lr=args.lr, weight_decay=0.0, parameter_groups=[]),
    epochs=args.epochs,
    eval_dataset=VisualNews("val", sample=32 if dbg else None),
    batch_size=args.batch_size,
    evaluator=VisualNewsEvaluator(meteor=False, rouge=False),
    prediction_args=dict(beam_search_spec=BeamSearchSpec(2, 16)),
    scheduler=scheduler,
    save_each_epoch=[args.epochs],
    num_workers=3
  )

  trainer.train(model, args.output_dir, override=args.override)


if __name__ == '__main__':
  main()
