"""Code from GPV-2 for saving FromParams objects to disk, used for model/trainer saving

AllenNLP recently added their own to_params approach, but there default implementation
does not work for some of our models so we stick with the GPV-2 version.
"""

import enum
import typing
from collections import OrderedDict
from inspect import signature, Parameter
from typing import Dict, Any, Type, Union

from allennlp.common import FromParams, Registrable
from allennlp.common.from_params import remove_optional
import numpy as np


def get_type_name(cls: Type[FromParams], super_cls: Type[Registrable]):
  for name in super_cls.list_available():
    if cls == super_cls.resolve_class_name(name)[0]:
      return name

  raise ValueError(f"Unable to find type name for {cls} from class {super_cls}, "
                   f"check if it was registered correctly")


def _has_args(anno):
  return (
      hasattr(anno, "__args__") and
      anno.__args__ is not None and
      len(anno.__args__) > 0
  )


def _is_fromparams(anno):
  if type(anno) not in {type, enum.EnumMeta}:
    # typing.* annotations do not work with issubclass, so fail them here since `FromParam`
    # objects will be annotated with actual types
    return False

  return issubclass(anno, FromParams)


def to_params_any(obj, annotation):
  """Convert and object with a type annotation to parameter or parameter dictionary"""

  if obj is None:
    # None is allowed for any annotation, so no type-checking required
    return obj

  obj_type = type(obj)
  annotation = remove_optional(annotation)
  origin = getattr(annotation, "__origin__", None)

  if origin is Union:

    if len(annotation.__args__) == 1:
      # This can happen after `remove_optional(Optional[str])` as well as
      # from user type annotations
      annotation = annotation.__args__[0]
    else:
      # Attempt to figure out what annotation is applicable, this is imperfect
      # since doing this in general is non-trivial, but we make a best-effort
      # attempt that works for simple cases.
      # We can fail if the union has multiple types that are only distinguished
      # by the generic type e.g., Union[List[int], List[str]]
      # Also if the union has superclasses of its own elements, e.g., Union[Number, float]
      # TODO there are a few ways to make this more general
      candidates = []
      for anno in annotation.__args__:
        if hasattr(anno, "__args__") and hasattr(anno, "__origin__") and anno.__args__ is not None:
          # Anno is a container with its own sub-types, hust check the top-level types
          if isinstance(obj, anno.__origin__):
            candidates.append(anno)
        else:
          if isinstance(obj, anno):
            candidates.append(anno)

      if len(candidates) == 0:
        raise ValueError(f"Object {obj} does not match any type annotation in Union {annotation}")
      if len(candidates) > 1:
        raise ValueError(f"Ambiguous Union {annotation} for object {obj}")
      annotation = candidates[0]

  # `FromParams` object, in which case we need an accurate annotation
  if isinstance(obj, FromParams):
    if annotation is typing.Any:
      raise ValueError(f"FromParams object {obj} needs type annotations")
    elif not _is_fromparams(annotation):
      raise ValueError(f"FromParams object {obj} has non-FromParams annotation {annotation}")
    return to_params(obj, annotation)
  elif _is_fromparams(annotation):
    raise ValueError(f"FromParams annotation {annotation} has non-FromParams object {obj}")

  # Base cases, no need to worry about annotations
  # note we allow incorrect typing here because I don't think it matters when loading the class
  if obj_type in {str, int, float, bool, np.integer, np.floating, np.ndarray, np.bool}:
    return obj

  # Collections, if there are type annotations, try to preserve them, since we will need
  # them if the collection contains `FromParams` classes
  # For the most part we trust clients to have correctly-typed containers
  elif obj_type in (list, set, frozenset): # Single arg containers
    if not _has_args(annotation):
      anno = typing.Any
    else:
      assert len(annotation.__args__) == 1, "Incorrect annotation"
      anno = annotation.__args__[0]
    return obj.__class__(to_params_any(x, anno) for x in obj)
  elif obj_type == tuple:
    if not _has_args(annotation):
      return obj.__class__(to_params_any(x, typing.Any) for x in obj)
    elif origin in (list, typing.List):  # Allow tuples for list objects
      assert len(annotation.__args__) == 1, "Incorrect annotation"
      anno = annotation.__args__[0]
      return obj.__class__(to_params_any(x, anno) for x in obj)
    else:
      if len(annotation.__args__) != len(obj):
        # TODO handle variable length tuple annotation
        raise ValueError()
      return obj.__class__(to_params_any(x, anno) for x, anno in zip(obj, annotation.__args__))
  elif obj_type in (dict, OrderedDict, Dict): # Two arg containers
    if not _has_args(annotation):
      k_anno, v_anno = typing.Any, typing.Any
    else:
      assert len(annotation.__args__) == 2, "Incorrect annotation"
      k_anno, v_anno = annotation.__args__
    output = obj.__class__()
    for k, v in obj.items():
      output[to_params_any(k, k_anno)] = to_params_any(v, v_anno)
    return output
  else:
    # Not a collection, base type, or FromParams, we can't convert it
    raise ValueError(f"Unable to convert {obj.__class__} to parameters")


def to_params(obj: FromParams, source_cls=None) -> Dict[str, Any]:
  """Tries to convert a `FromParams` object to its parameter dictionary.

  This requires `obj` to store the parameters to __init__ as attributes with the
  corresponding parameter name, or to to provided a `to_params` method. These attributes
  should themselves be basic python types, or other FromParams classes.
  Any `FromParams` instances found needs to have accurate type annotations
  """
  cls = obj.__class__

  if (
      hasattr(obj, "to_params") and
      cls._to_params != Registrable._to_params and
      cls._to_params != FromParams._to_params
  ):
    # If the object has overridden the default to_params method, use that
    args = obj.to_params().as_dict(quiet=True)

  else:
    init = cls.__init__
    if init is object.__init__:
      args = {}  # No init args
    else:
      init_signature = signature(init)
      for param in init_signature.parameters.values():
        if param.kind != Parameter.POSITIONAL_OR_KEYWORD:
          raise NotImplementedError(cls.__name__ + " has **kwargs or *args in __init__")
      param_names = [p for p in init_signature.parameters.keys() if p != "self"]
      args = {}
      for name in param_names:
        if not hasattr(obj, name):
          raise ValueError(cls.__name__ + " did not store parameter " + name)
        val = getattr(obj, name)
        annotation = init_signature.parameters[name].annotation
        args[name] = to_params_any(val, annotation)

  if source_cls is not None and source_cls != cls:
    args["type"] = get_type_name(cls, source_cls)
  return args
