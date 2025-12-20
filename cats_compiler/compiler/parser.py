from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, UNIQUE, verify
from typing import List, Optional, Dict, Any, Union

@verify(UNIQUE)
class SplitType(Enum):
  FloatFeature = "FloatFeature"
  OneHotFeature = "OneHotFeature"
  OnlineCtr = "OnlineCtr"  # not supported yet

@dataclass(frozen=True)
class Feature:
  index: int
  is_categorical: bool

@dataclass(frozen=True)
class Split:
  type: SplitType
  feature_index: int
  threshold: Optional[float] = None
  cat_value: Optional[int] = None

@dataclass
class Tree:
  depth: int
  splits: List[Split]
  leaf_values: List[float]

  def leaf_index(self, features_bin: List[int]) -> int:
    """Compute leaf index for oblivious (symmetric) trees from pre-binarized features.

    features_bin: for FloatFeature it's 1 if feature > threshold else 0;
                  for OneHotFeature it's 1 if cat_value matches else 0.
    """
    idx = 0
    for bit in features_bin[:self.depth]: idx = (idx << 1) | (1 if bit else 0)
    return idx


@dataclass
class Model:
  features: List[Feature]
  trees: List[Tree]
  scale: float = 1.0
  bias: float = 0.0

  @staticmethod
  def from_catboost_json(data: Dict[str, Any]) -> "Model":
    """Parse a CatBoost JSON dump (format='json') into Model IR.

    Only supports oblivious (symmetric) trees and FloatFeature/OneHotFeature splits.
    """
    features: List[Feature] = []

    features_info = data.get("features_info") or {}
    float_features = features_info.get("float_features") or []
    categorical_features = features_info.get("categorical_features") or []

    flat_to_is_cat: Dict[int, bool] = {}
    for f in float_features:
      flat_idx = f.get("flat_feature_index") or f.get("feature_index")
      if flat_idx is not None:
        flat_to_is_cat[int(flat_idx)] = False
    for f in categorical_features:
      flat_idx = f.get("flat_feature_index") or f.get("feature_index")
      if flat_idx is not None:
        flat_to_is_cat[int(flat_idx)] = True

    # Build a dense feature list (if indices are sparse, we fill gaps as numerical)
    if flat_to_is_cat:
      max_idx = max(flat_to_is_cat.keys())
      for i in range(max_idx + 1):
        features.append(Feature(index=i, is_categorical=flat_to_is_cat.get(i, False)))

    trees: List[Tree] = []
    oblivious_trees = data.get("oblivious_trees") or []
    for t in oblivious_trees:
      depth = int(t.get("tree_depth", 0))
      splits_json = t.get("splits") or []
      leaf_values = [float(v) for v in (t.get("leaf_values") or [])]

      splits: List[Split] = []
      for s in splits_json:
        stype = s.get("split_type") or s.get("type")
        if stype == "FloatFeature":
          splits.append(
            Split(
              type=SplitType.FloatFeature,
              feature_index=int(s.get("float_feature_index") or s.get("feature_index")),
              threshold=float(s.get("border")),
            )
          )
        elif stype == "OneHotFeature":
          splits.append(
            Split(
              type=SplitType.OneHotFeature,
              feature_index=int(s.get("cat_feature_index") or s.get("feature_index")),
              cat_value=int(s.get("value") or s.get("value_index") or 0),
            )
          )
        else:
          raise ValueError(f"Unsupported split type: {stype}")

      if depth and (1 << depth) != len(leaf_values):
        # Some CatBoost dumps may store leaf_values concatenated for all trees; but in standard json this holds per tree
        raise ValueError(
          f"Leaf values count {len(leaf_values)} doesn't match 2^depth ({1<<depth}) for a tree"
        )

      trees.append(Tree(depth=depth, splits=splits, leaf_values=leaf_values))

    scale = float((data.get("scale_and_bias") or {}).get("scale", 1.0))
    bias = float((data.get("scale_and_bias") or {}).get("bias", 0.0))

    return Model(features=features, trees=trees, scale=scale, bias=bias)
