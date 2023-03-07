import logging
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any, Callable, List, Dict, Tuple, Union, Optional

import numpy as np
import clip
import torch
from PIL import Image
from allennlp.common import Registrable, Params
from torch import nn
from transformers import T5ForConditionalGeneration, AutoTokenizer, AutoConfig, T5Config, \
  T5Tokenizer
from transformers.modeling_outputs import Seq2SeqLMOutput

from close.data.coco_captioning import CaptioningExample
from close.data.visual_entailment import VisualEntailmentExample
from l2v.data.visual_news import VisualNewsExample
from close.data.vqa_v2 import VqaExample
from close.model.layers import Layer
from close.model.model import Model, ExampleOutput, BeamSearchSpec
from close.train.allennlp_beamsearch import t5_initialize_decoding
from close.utils import image_utils, pytorch_utils

from close.utils.pytorch_utils import replace_parameters

CLIP_DIMS = {
  "ViT-B/32": 512,
  "ViT-L/14": 768,
  "RN101": 512,
  "RN50": 1024,
  "RN50x4": 640,
  "RN50x16": 768,
  "RN50x64": 1024
}


@Layer.register("linear")
class EmbeddingTokenizer(Layer):

  def __init__(self, n_tokens: int=4, n_constant: int=0):
    super().__init__()
    self.n_tokens = n_tokens
    self.n_constant = n_constant

  def init(self, t5_dim, clip_dim):
    self.t5_dim = t5_dim
    self.lin = nn.Linear(clip_dim, t5_dim*self.n_tokens)
    if self.n_constant:
      self.constant_tokens = nn.Parameter(torch.zeros(self.n_constant, t5_dim))

  def forward(self, clip_features):
    seq = self.lin(clip_features).reshape(-1, self.n_tokens, self.t5_dim)
    if self.n_constant:
      seq = torch.cat([self.constant_tokens.unsqueeze(0).tile(seq.size(0), 1, 1), seq], 1)
    return seq


@dataclass
class TrainingExample:
  image_id: Optional[str] = None
  target_text: Union[List[str], None] = None
  input_text: Optional[str] = None
  image_text: Union[str, List[str], None] = None
  example_id: Optional[str] = None


@dataclass
class Collate:
  tokenizer: Any
  pre: Any
  encode_image: bool

  def __call__(self, batch: List[TrainingExample]):
    out = {}

    # Encode the target text, To support examples with multiple target texts,
    # track text->example_num. mapping This lets us efficiently handle this examples
    # by avoiding re-encoding the context for each target
    if batch[0].target_text is not None:
      texts = []
      mapping = []
      for batch_ix, x in enumerate(batch):
        texts += x.target_text
        mapping += [batch_ix]*len(x.target_text)
      out["target_mapping"] = torch.as_tensor(mapping, dtype=torch.long)
      labels = self.tokenizer(
        texts, return_tensors='pt', padding=True, truncation=True)
      out["target_ids"] = labels["input_ids"]
    else:
      out["target_ids"] = None   # For testing

    # Encode any additional text context (e.g, question, hypothesis)
    if batch[0].input_text is not None:
      texts = [x.input_text for x in batch]
      labels = self.tokenizer(
        texts, return_tensors='pt', padding=True, truncation=True)
      out["input_ids"] = labels["input_ids"]
      out["input_attention_mask"] = labels["attention_mask"]

    # Encode the image or text input
    if self.encode_image:
      images = []
      for ex in batch:
        with Image.open(image_utils.get_image_file(ex.image_id)) as f:
          images.append(self.pre(f))
      out["clip_images"] = torch.stack(images, 0)
    elif isinstance(batch[0].image_text, str):
      out["clip_text"] = clip.tokenize([x.image_text for x in batch], truncate=True)
      out["clip_images"] = None
    else:
      # Multiple image target captions that will be averaged to get an input vector
      texts = []
      mapping = []
      for batch_ix, x in enumerate(batch):
        if isinstance(x.image_text, str):
          texts.append(x.image_text)
          mapping.append(batch_ix)
        else:
          texts += x.image_text
          mapping += [batch_ix]*len(x.image_text)
      out["clip_text"] = clip.tokenize(texts)
      out["clip_images"] = None
      out["input_average_mapping"] = torch.as_tensor(mapping, dtype=torch.long)
    return out


@Model.register("clip-t5")
class ClipT5Model(Model):

  @classmethod
  def from_params(
    cls, params: Params, constructor_to_call=None,
    constructor_to_inspect=None, **extras
  ):
    return super().from_params(params, constructor_to_call, constructor_to_inspect, **extras)

  def __init__(
      self, clip_model: str, t5_model_name: str, adapter: Layer,
      language_shift: Layer=None, openai_clip=None, train_on_l: bool=True,
      lowercase_target=False, caption_mode="other-target", average_vqa_caption=True):
    super().__init__()
    self.openai_clip = openai_clip
    self.language_shift = language_shift
    self.lowercase_target = lowercase_target
    self.clip_model = clip_model
    self.t5_model_name = t5_model_name
    self.adapter = adapter
    self.train_on_l = train_on_l
    self.caption_mode = caption_mode
    self.average_vqa_caption = average_vqa_caption

    # Set during init
    self._clip_model = None
    self._clip_pre = None
    self._t5_model = None
    self.tokenizer = None
    self.image_id_to_ix = None

    # Prediction args
    self.beam_search_spec = None

  def initialize(self, load_params=True):
    """Initialize the model by constructing the need pytorch sub-modules

    if `load_params` is true the pre-trained weights will be loaded, if false
    the parameters will random (usually because a state_dict will be loaded)
    """
    if self.openai_clip:
      import open_clip
      logging.info(f"Loading clip {self.clip_model}/{self.openai_clip}...")
      model, _, preprocess = open_clip.create_model_and_transforms(
        self.clip_model, pretrained=self.openai_clip)
    else:
      logging.info(f"Loading clip {self.clip_model}...")
      model, preprocess = clip.load(self.clip_model)

    # Store the CLIP parameters as non-persistent buffers so it
    # doesn't take up space in the state_dict
    replace_parameters(model, False)

    clip_dim = CLIP_DIMS[self.clip_model]
    self._clip_pre = preprocess
    self._clip_model = model
    for param in self._clip_model.parameters():
      param.requires_grad = False

    logging.info(f"Loading T5 {self.t5_model_name}...")
    if load_params:
      self._t5_model = T5ForConditionalGeneration.from_pretrained(self.t5_model_name)
    else:
      self._t5_model = T5ForConditionalGeneration(AutoConfig.from_pretrained(self.t5_model_name))
    t5_dim = self._t5_model.encoder.config.d_model

    print("DEBUG")
    self.tokenizer = T5Tokenizer.from_pretrained(self.t5_model_name, local_files_only=True)
    self.adapter.init(t5_dim, clip_dim)

  def get_collate(self, is_train=False) -> Callable[[List], Dict[str, Any]]:
    """Get the collate function that should be used with this model"""
    if is_train:
      encode_image = not self.train_on_l
    else:
      encode_image = True
    return Collate(self.tokenizer, self._clip_pre, encode_image)

  def preprocess_example_train(self, ex) -> List[TrainingExample]:
    """Transform an input example into a general-purpose format we can collate"""

    if isinstance(ex, CaptioningExample):
      if not self.train_on_l:
        out = [TrainingExample(ex.image_id, ex.captions, image_text=None)]
      elif self.caption_mode == "1to1":
        out = [TrainingExample(ex.image_id, [x], image_text=x) for x in ex.captions]
      elif self.caption_mode == "other-target":
        targets = ex.captions
        out = []
        for i, text in enumerate(targets):
          out.append(TrainingExample(ex.image_id, targets[:i] + targets[i+1:], image_text=text))
      else:
        raise NotImplementedError(self.caption_model)

    elif isinstance(ex, VisualEntailmentExample):
      out = [TrainingExample(
        ex.image_id, [ex.label], ex.hypothesis, image_text=ex.premise, example_id=ex.example_id)]

    elif isinstance(ex, VqaExample):
      target_text = []
      if isinstance(ex.answers, Counter):
        # Train on all answers that either occur more than 3 times, or are as common as the
        # most common answer
        on = None
        for w, c in ex.answers.most_common():
          if on is None:
            target_text.append(w)
            on = c
          elif c == on or c >= 3:
            target_text.append(w)
          else:
            break
      else:
        assert isinstance(ex.answers, str)
        target_text = [ex.answers]
      if isinstance(ex.image_text, str) or self.average_vqa_caption:
        out = [TrainingExample(ex.image_id, target_text,
                               ex.question, ex.image_text, example_id=ex.example_id)]
      else:
        out = [TrainingExample(ex.image_id, target_text, ex.question, x, example_id=ex.example_id)
               for x in ex.image_text]
        
    elif isinstance(ex, VisualNewsExample):
      extract_i = (self.train_on_l in {"never", "both"} or
                   (self.train_on_l in {"optional", "skip-lang"} and ex.image_id is not None))

      out = [
        TrainingExample(
          example_id=ex.example_id,
          image_id=ex.image_id if extract_i else None, 
          input_text=ex.article,
          image_text=ex.caption,
          target_text=[ex.caption])
      ]
    else:
      raise NotImplementedError()

    if self.lowercase_target:
      for ex in out:
        ex.target_text = [x.lower() for x in ex.target_text]
    return out

  def preprocess_example(self, example) -> TrainingExample:
    """Preprocess a train example"""

    if isinstance(example, CaptioningExample):
      if example.captions:
        # In case we are testing with text input
        cap = example.captions[np.random.randint(0, len(example.captions))]
      else:
        cap = None
      return TrainingExample(example_id=example.example_id, image_id=example.image_id,
                             target_text=None, input_text=None, image_text=cap)
    elif isinstance(example, VqaExample):
      return TrainingExample(example_id=example.example_id, image_id=example.image_id,
                             target_text=None, input_text=example.question,
                             image_text=example.image_text)

    elif isinstance(example, VisualEntailmentExample):
      return TrainingExample(example_id=example.example_id, image_id=example.image_id,
                             input_text=example.hypothesis,
                             target_text=None, image_text=example.premise)
    elif isinstance(example, VisualNewsExample):
      return TrainingExample(example_id=example.example_id,
                             image_id=example.image_id,
                             input_text=example.article,
                             image_text=example.caption,
                             target_text=None)
    else:
      raise NotImplementedError()

  def _encode(self, clip_images, clip_text, input_ids, input_attention_mask,
              input_average_mapping=None):
    if clip_images is not None:
      assert clip_text is None
      with torch.no_grad():
        image_fe = self._clip_model.encode_image(clip_images)
      image_fe = image_fe.float()
      image_fe = image_fe / image_fe.norm(dim=-1, keepdim=True)
      clip_features = image_fe

    else:
      assert clip_images is None
      with torch.no_grad():
        text_fe = self._clip_model.encode_text(clip_text)
      text_fe = text_fe.float()
      if input_average_mapping is not None:
        text_fe = pytorch_utils.segment_mean(text_fe, input_average_mapping)
      text_fe = text_fe / text_fe.norm(dim=-1, keepdim=True)
      text_fe = self.language_shift(text_fe)
      clip_features = text_fe

    clip_tokens = self.adapter(clip_features)

    if input_ids is not None:
      input_embed = self._t5_model.shared(input_ids)
      input_embed, input_mask = pytorch_utils.concat_masked_sequences(
        clip_tokens, input_embed, input_attention_mask)
    else:
      input_embed = clip_tokens
      input_mask = None
    encoding = self._t5_model.encoder(
      inputs_embeds=input_embed,
      return_dict=True
    ).last_hidden_state
    return encoding, input_mask

  def forward(
      self, clip_images, clip_text, target_ids, target_mapping=None,
      input_average_mapping=None, input_ids=None, input_attention_mask=None
  ) -> Tuple[torch.Tensor, Dict[str, float]]:
    target_ids = target_ids.masked_fill(
      target_ids == self.tokenizer.pad_token_id, -100)
    encoder_out, input_mask = self._encode(clip_images, clip_text, input_ids,
                                           input_attention_mask, input_average_mapping)

    if target_mapping is not None:
      encoder_out = encoder_out[target_mapping]
      if input_mask is not None:
        input_mask = input_mask[target_mapping]

    out: Seq2SeqLMOutput = self._t5_model(
      encoder_outputs=(encoder_out, ),
      attention_mask=input_mask,
      labels=target_ids,
      return_dict=True
    )
    return out.loss, {}

  def set_prediction_args(self, beam_search_spec: BeamSearchSpec):
    self.beam_search_spec = beam_search_spec

  def predict(self, clip_images=None, clip_text=None, target_ids=None, target_mapping=None,
              input_ids=None, input_attention_mask=None):
    enc, input_mask = self._encode(clip_images, clip_text, input_ids, input_attention_mask)

    bs = self.beam_search_spec.build(self.tokenizer.eos_token_id)
    decode_init = t5_initialize_decoding(
      self.tokenizer, self._t5_model, enc, input_mask)
    input_ids, logprobs = bs.search(*decode_init)

    logprobs = logprobs.cpu().numpy()
    input_ids = input_ids.cpu().numpy()

    out_text = []
    for batch in range(len(input_ids)):
      text = [self.tokenizer.decode(x, skip_special_tokens=True) for x in input_ids[batch]]
      out_text.append(text)

    return [ExampleOutput(txt, p.tolist()) for txt, p in zip(out_text, logprobs)]

