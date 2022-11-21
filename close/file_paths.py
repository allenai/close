from os import mkdir
from os.path import join, dirname, expanduser, exists

DATA_DIR = expanduser("~/data-dbg")

COCO_SOURCE = join(DATA_DIR, "coco")
COCO_ANNOTATIONS = join(COCO_SOURCE, "annotations")
COCO_IMAGES = join(COCO_SOURCE, "images")

VQAE = join(DATA_DIR, "vqa-e")

SNLI_VE_HOME = join(DATA_DIR, "SNLI_VE")
FLICKER30K = join(DATA_DIR, "SNLI_VE", "Flickr30K", "flickr30k_images")

VQA_ANNOTATIONS = join(DATA_DIR, "vqa")

ADAPTER_SOURCE = join(DATA_DIR, "adapters")