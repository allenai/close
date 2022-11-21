import gzip
import logging
import tarfile
import tempfile
import zipfile
from os import listdir, makedirs
from os.path import dirname, exists, join

import requests
from tqdm import tqdm

from close import file_paths
from close.utils import py_utils


def ensure_dir_exists(filename):
  """Make sure the parent directory of `filename` exists"""
  makedirs(dirname(filename), exist_ok=True)


def download_to_file(url, output_file, pbar=False, unzip=False):
  """Download `url` to `output_file`"""
  logging.info(f"Downloading file from {url} to {output_file}")
  ensure_dir_exists(output_file)

  if not pbar:
    with requests.get(url) as r:
      r.raise_for_status()
      content = r.content
      if unzip:
        content = gzip.decompress(content)
      with open(output_file, 'wb') as f:
        f.write(content)
  else:
    if unzip:
      raise NotImplementedError()
    with requests.get(url, stream=True) as r:
      r.raise_for_status()
      with open(output_file, 'wb') as f:
        _write_to_stream(r, f, True)


def download_zip(url, source, progress_bar=True):
  """Download zip file at `url` and extract to `source`"""
  # Download to a temp file to ensure we
  # don't eat a lot of RAM with downloading a large file
  with tempfile.TemporaryFile() as tmp_f:
    with requests.get(url, stream=True) as r:
      _write_to_stream(r, tmp_f, progress_bar)

    logging.info("Extracting to %s...." % source)
    makedirs(source, exist_ok=True)
    with zipfile.ZipFile(tmp_f) as f:
      f.extractall(source)


def download_tar(url, source, progress_bar=True):
  """Download tar file at `url` and extract to `source`"""
  with tempfile.TemporaryFile() as tmp_f:
    with requests.get(url, stream=True) as r:
      _write_to_stream(r, tmp_f, progress_bar)

    logging.info("Extracting to %s...." % source)
    makedirs(source, exist_ok=True)
    tmp_f.seek(0)
    with tarfile.open(fileobj=tmp_f) as f:
      f.extractall(source)


DRIVE_URL = "https://docs.google.com/uc?export=download"


def download_from_drive(file_id, output_file, progress_bar=False):
  """Download the public google drive file `file_id` to `output_file`"""
  ensure_dir_exists(output_file)

  session = requests.Session()

  response = session.get(DRIVE_URL, params={'id': file_id}, stream=True)

  # Check to see if we need to send a second, confirm, request
  # https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
  for key, value in response.cookies.items():
    if key.startswith('download_warning'):
      params = {'id': file_id, 'confirm': value}
      response = session.get(DRIVE_URL, params=params, stream=True)
      break

  with open(output_file, "wb") as f:
    _write_to_stream(response, f, progress_bar)
  response.close()


def _write_to_stream(response, output_fh, progress_bar=True, chunk_size=32768):
  """Write streaming `response` to `output_fs` in chunks"""
  response.raise_for_status()
  if progress_bar:
    content_len = response.headers.get("Content-Length")
    if content_len is not None:
      total = int(content_len)
    else:
      total = None
    pbar = tqdm(desc="downloading", total=total, ncols=100, unit="b", unit_scale=True)
  else:
    pbar = None

  cur_total = 0
  for chunk in response.iter_content(chunk_size=chunk_size):
    if chunk:  # filter out keep-alive new chunks
      if pbar is not None:
        cur_total += len(chunk)
        next_value = cur_total
        pbar.update(next_value - pbar.n)
      output_fh.write(chunk)

  if pbar is not None:
    if pbar.total is not None:
      pbar.update(pbar.total - pbar.n)
    pbar.close()


VE_SRC = "https://storage.googleapis.com/allennlp-public-data/snli-ve"


def download_snli_ve():
  if not exists(file_paths.SNLI_VE_HOME):
    for file in ["snli_ve_train.jsonl", "snli_ve_dev.jsonl", "snli_ve_test.jsonl"]:
      download_to_file(f"{VE_SRC}/{file}.gz", join(file_paths.SNLI_VE_HOME, file), unzip=True)

  if not exists(file_paths.FLICKER30K):
    logging.info(f"Downloading {VE_SRC}/flickr30k_images.tar.gz")
    download_tar(
      f"{VE_SRC}/flickr30k_images.tar.gz",
      dirname(file_paths.FLICKER30K)
    )


VQAE_FILES = {
  "1CXogPObRixI1iR51T2px-Q75jdnhByCX":  "VQA-E_train_set.json",
  "12e8Px79J4lOT0NBUe2JVzTjbgfRy06qY":  "VQA-E_val_set.json",
}


def download_vqae():
  for file_id, filename in VQAE_FILES.items():
    filename = join(file_paths.VQAE, filename)
    if not exists(filename):
      logging.info(f"Downloading {filename}")
      download_from_drive(file_id, join(file_paths.VQAE, filename))


VQA_FILES = {
  "questions": {
    "train": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip",
    "val": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
    "test": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip"
  },
  "annotations": {
    "train": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip",
    "val": "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip"
  }
}


def download_vqa_annotations():
  if exists(file_paths.VQA_ANNOTATIONS):
    return
  for kind, files in VQA_FILES.items():
    for split, file in files.items():
      logging.info(f"Downloading {file}")
      download_zip(file, file_paths.VQA_ANNOTATIONS, False)


COCO_ANNO = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
COCO_IMAGES = {
  "val2014": "http://images.cocodataset.org/zips/val2014.zip",
  "test2014": "http://images.cocodataset.org/zips/test2014.zip",
  "train2014": "http://images.cocodataset.org/zips/train2014.zip",
  "test2015": "http://images.cocodataset.org/zips/test2015.zip",
}


def download_coco():
  if not exists(join(file_paths.COCO_SOURCE, "annotations")):
    logging.info(f"Downloading {COCO_ANNO}")
    download_zip(COCO_ANNO, file_paths.COCO_SOURCE, True)

  for k, url in COCO_IMAGES.items():
    if not exists(join(file_paths.COCO_IMAGES, k)):
      logging.info(f"Downloading {url}")
      download_zip(url, file_paths.COCO_IMAGES, True)


ADAPTER_HOME = "https://ai2-prior-close.s3.us-west-2.amazonaws.com/adapters/"

adapter_paths = [
  "cc3m-cov-diff.pkl",
  "cc3m-linear-v1.pkl",
  "cc3m-mean-diff.pkl",
  "kp-restval-cov-diff.pkl",
  "kp-restval-linear-v1.pkl",
  "kp-restval-mean-diff.pkl",
]


def download_adapters():
  for file_name in adapter_paths:
    output = join(file_paths.ADAPTER_SOURCE, file_name)
    if not exists(output):
      logging.info(f"Downloading {file_name} to {output}")
      download_to_file(join(ADAPTER_HOME, file_name), output)


KP_SOURCE = "http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip"


def download_kp():
  if not exists(join(file_paths.COCO_SOURCE, "dataset_coco.json")):
    download_zip(KP_SOURCE, file_paths.COCO_SOURCE)


def main():
  py_utils.add_stdout_logger()
  download_coco()
  download_kp()
  download_vqae()
  download_snli_ve()
  download_vqa_annotations()
  download_adapters()


if __name__ == '__main__':
  main()