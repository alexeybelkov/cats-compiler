from typing import Any
from enum import Enum, verify, UNIQUE
import deepcopy


RawSplitT = dict[str, str|int|float]


@verify(UNIQUE)
class SplitType(Enum):
  FloatFeature = 0
  OneHotFeature = 1
  OnlineCtr = 2
  def get_type(self, raw_split: RawSplitT) -> "SplitType":
    if "float_feature_index" in raw_split:
      return self.FloatFeature
    else:
      return self.OneHotFeature
  def get_index_name(self, raw_split: RawSplitT) -> str:
    return _INDEX_NAMES.intersection(raw_split.keys()).pop()


class Leaf:
  def __init__(self, value: float, weight: float):
    self.value = value
    self.weight = weight
  def __add__(self, rhs: float) -> float:
    return self.value * self.weight + rhs.value * rhs.weight


class Split:
  def __init__(self, border: float, feature_index: int, split_index: int, split_type: SplitType):
    self.border = border
    self.feature_index = feature_index
    self.split_index = split_index
    self.split_type = split_type
    self.typed_feature_index: int | None = None


class Tree:
  def __init__(self, leaf_values: list[float], leaf_weights: list[float], raw_splits: list[dict[str, float|str]]):
    self.splits: list[Split] = []
    for split in raw_splits:
      split_type = SplitType.get_type(split)
      self.splits.append(Split(split['border'], split['']))
    self.leaf_values = leaf_values
    self.leaf_weights = leaf_weights
  @classmethod
  def from_dict(cls, tree: dict[str, Any]) -> 'Tree':
    return cls(tree['leaf_values'], tree['leaf_weights'], tree['splits'])
  @classmethod
  def from_tree(cls, tree: 'Tree') -> 'Tree':
    return deepcopy.copy(tree)
  def __repr__(self) -> str:
    return super().__repr__()
