"""Provides functions to use T5 in allennlp's BeamSearch

We use this instead of transformer's beam search mostly for legacy reasons since that is
what the GPV-2 models used
"""

import torch
from torch.nn import functional as F

from close.utils import py_utils


def t5_initialize_decoding(tokenizer, model, encoder_out, encoder_mask, post_process=None):
  batch_size = encoder_out.size(0)
  device = encoder_out.device
  initial_state = dict(
    encoder_mask=encoder_mask,
    encoder_outputs=encoder_out
  )

  def _decode_step(predictions, prev_state, time_step):
    return _t5_decoding_step(model, predictions, prev_state, post_process, time_step)

  initial_out = torch.full(
    (batch_size,), tokenizer.pad_token_id, dtype=torch.long, device=device)

  return initial_out, initial_state, _decode_step


def _t5_decoding_step(model, predictions, state, post_process, time_step):
  past = py_utils.flat_to_nested_struct({k: v.contiguous() for k, v in state.items()
                                         if isinstance(k, tuple)})
  model_inputs = model.prepare_inputs_for_generation(
    predictions.unsqueeze(1),
    past=past, attention_mask=state["encoder_mask"],
    encoder_outputs=(state["encoder_outputs"],),
    use_cache=True)
  out = model(**model_inputs, return_dict=True)
  logits = out.logits

  logits = logits.squeeze(1)
  logits = F.log_softmax(logits, -1)

  if post_process is not None:
    logits = post_process(logits, model_inputs, time_step)

  next_state = dict(
    encoder_mask=state["encoder_mask"],
    encoder_outputs=state["encoder_outputs"],
  )
  py_utils.nested_struct_to_flat(out.past_key_values, cur_dict=next_state)
  return logits, next_state
