from os import listdir
from os.path import join

from close import file_paths

_IMAGE_ID_TO_SIZE_MAP = {}

IMAGE_SOURCE_MAP = {
  "coco": file_paths.COCO_IMAGES,
  "flicker30k": file_paths.FLICKER30K,
}


def get_image_file(image_id) -> str:
  """Returns the filepath of an image corresponding to an input image id

  To support multiple datasets, we prefix image_ids with "source/"
  We use this extra level of indirection instead of the using filepaths directly to allow
  file-system independent image_ids
  """
  source, key = image_id.split("/", 1)
  if source in IMAGE_SOURCE_MAP:
    return join(IMAGE_SOURCE_MAP[source], key)
  raise ValueError(f"Unknown image id {image_id}")


def get_coco_image_id(subset, image_id):
  """Turns COCO image_id into a COCO filepath"""
  return f'coco/{subset}/COCO_{subset}_{str(image_id).zfill(12)}.jpg'


_IMAGE_TO_SUBSETS = None


def get_coco_subset(image_id: int) -> str:
  """Get the COCO subset an image belongs"""
  global _IMAGE_TO_SUBSETS
  if _IMAGE_TO_SUBSETS is None:
    _IMAGE_TO_SUBSETS = {}
    for subset in ["train2014", "val2014"]:
      for image_file in listdir(join(file_paths.COCO_IMAGES, subset)):
        image_id = int(image_file.split("_")[-1].split(".")[0])
        _IMAGE_TO_SUBSETS[image_id] = subset
  return _IMAGE_TO_SUBSETS[image_id]


def get_coco_id_from_int_id(image_id: int) -> str:
  """Get a COCO image id from an int image"""
  return get_coco_image_id(get_coco_subset(image_id), image_id)

