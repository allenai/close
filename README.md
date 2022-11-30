# CLIP Cross-Modality Transfer
Code for our paper: [*I can't believe there's no Images!* Learning Visual Tasks Using only Language Data](https://arxiv.org/abs/2211.09778).

This project trains models on pure-text data and then shows they can applied to the same tasks
with visual inputs instead of text, thus demonstrating zero-shot cross-modal transfer.
This is done by using the shared semantic embedding space of contrastive vision and language models.

More details can be found on our [webpage](https://prior.allenai.org/projects/close).

## Installation
Install [pytorch](https://pytorch.org/), we have tested this code with torch 1.10.1 and 1.11.0
Then install the other requirements with:

```
pip install -r requirements.txt
```

Finally download the needed datasets, data should be saved in the paths stored in close/file_paths.py.
The data can be downloaded automatically using this script:

```
python close/download.py
```

This will download about 45G of data. Data can also be manually downloaded these sources:

### COCO data:
The 2014 train and validation images for [COCO](https://cocodataset.org/#download) should be put in ~/data/coco/images
The annotations should be saved to ~/data/coco/annotations.

### Visual Entailment:
The data build from [here](https://github.com/necla-ml/SNLI-VE).
By default the annotations should be in ~/data/SNLI_VE 
and the images should be in ~/data/SNLI_VE/Flickr30K

### VQA-E:
Download the files from [here](https://github.com/liqing-ustc/VQA-E),
by default the files should be put into ~/data/vqa-e

### COCO Captions:
We use the Karpathy Split found [here](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip)
The coco_dataset.json file should be put into ~/data/coco.

### Adapters 
The linear and covariance adapters we used in our ablations can be downloaded directly from
AWS, see `download_adapters` in `coco/close/download.py` 


### VQA:
VQA annotation from [here](https://visualqa.org/download.html), which should be saved ~/data/vqa2

## Training
Training is done with cross/experiments/train.py,

``
python close/experiments/train.py --data {vqa|vqa-e|ve|s-cap|m-cap} --output_dir path/to/output/dir
``

The `data` controls which dataset to train on, s-cap is captioning in the single captioning
setting and m-cap is the multiple captioning setting.

The script will use our default values, see the command line args for how to change the 
parameters.  

## Evaluation
The evaluations can be done with `eval.py`, for example:

```
python coco/experiments/eval.py path/to/model evqa --output_name default 
```

## Trained Models
Each model includes a `model.json` and `state-ep8.pth` file, the E-VQA model can be downloaded like this:

```
mkdir model
mkdir model/r0
wget https://ai2-prior-close.s3.us-west-2.amazonaws.com/models/evqa/model.json -O model/model.json
wget https://ai2-prior-close.s3.us-west-2.amazonaws.com/models/evqa/r0/state-ep8.pth -O model/r0/state-ep8.pth
```

Other models can be downloaded by replace `evqa` with these names:

- s-cap: Captioning (Single): 
- m-cap: Captioning (Multiple): 
- vqa: VQA (note trained on train+val):
- evqa: EVQA:
- ve: Visual Entailment:

