"""ASCII visualizations for CatBoost oblivious (symmetric) trees.

Standalone: every function takes primitive data (splits / leaf_values /
leaf_weights), so it does not depend on any particular Tree class. Wire it into
a Tree like:

    from cats_compiler.compiler.ast.viz import tree_repr, tree_render
    class Tree:
        def __repr__(self):
            return tree_repr(self.splits, self.leaf_values, self.leaf_weights, self.dim)

Oblivious tree convention: depth == len(splits); split d is shared by every node
at depth d; a leaf has a `depth`-bit index whose bit d is
(feature[splits[d]] > border[d])  — the LSB-at-depth-0 form used by the C++
export (index |= cmp << d). leaf_values is leaf-major: leaf i's `dim` outputs are
leaf_values[i*dim : (i+1)*dim].
"""
from typing import Any, Sequence

_SPARK = "▁▂▃▄▅▆▇█"


# ── split formatting ───────────────────────────────────────────────────────────
def _split_label(split: Any) -> str:
    """Accepts either a JSON dict or an object with .feature_index/.border."""
    if isinstance(split, dict):
        fi = split.get("float_feature_index", split.get("feature_index", -1))
        border = split["border"]
        si = split.get("split_index")
    else:
        fi = getattr(split, "feature_index", -1)
        border = split.border
        si = getattr(split, "split_index", None)
    tag = f"   [split {si}]" if si is not None else ""
    return f"f{fi} > {border:+.4f}{tag}"


def _feature_index(split: Any) -> int:
    if isinstance(split, dict):
        return split.get("float_feature_index", split.get("feature_index", -1))
    return getattr(split, "feature_index", -1)


def _border(split: Any) -> float:
    return split["border"] if isinstance(split, dict) else split.border


# ── leaf helpers ───────────────────────────────────────────────────────────────
def _leaf_scalar(values: Sequence[float], idx: int, dim: int) -> float:
    if dim == 1:
        return values[idx]
    chunk = values[idx * dim:(idx + 1) * dim]
    return sum(chunk) / len(chunk)


def _leaf_str(values: Sequence[float], weights: Sequence[float], idx: int, dim: int, depth: int) -> str:
    bits = format(idx, f"0{depth}b")
    w = weights[idx] if idx < len(weights) else 0
    if dim == 1:
        val = f"{values[idx]:+.5f}"
    else:
        val = "[" + ", ".join(f"{x:+.3f}" for x in values[idx * dim:(idx + 1) * dim]) + "]"
    return f"leaf[{bits}] = {val}  (w={w:g})"


def _sparkline(values: Sequence[float], num_leaves: int, dim: int) -> str:
    vals = [_leaf_scalar(values, i, dim) for i in range(num_leaves)]
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return _SPARK[0] * len(vals)
    return "".join(_SPARK[int((v - lo) / (hi - lo) * (len(_SPARK) - 1))] for v in vals)


# ── public API ───────────────────────────────────────────────────────────────
def tree_repr(splits: Sequence[Any], leaf_values: Sequence[float],
              leaf_weights: Sequence[float], dim: int = 1) -> str:
    """Compact view: one line per split level + a leaf-value sparkline.
    Scales fine to depth 6 (the oblivious structure is `depth` decisions, not 2^depth)."""
    depth = len(splits)
    num_leaves = 1 << depth
    lines = [f"ObliviousTree(depth={depth}, dim={dim}, leaves={num_leaves})"]
    for d, s in enumerate(splits):
        lines.append(f"  d{d}: {_split_label(s)}")
    vals = [_leaf_scalar(leaf_values, i, dim) for i in range(num_leaves)]
    lines.append(f"  leaves {_sparkline(leaf_values, num_leaves, dim)}")
    lines.append(f"         min {min(vals):+.4f}  max {max(vals):+.4f}")
    return "\n".join(lines)


def tree_render(splits: Sequence[Any], leaf_values: Sequence[float],
                leaf_weights: Sequence[float], dim: int = 1) -> str:
    """Full expanded binary tree. Left edge = '≤' (bit 0), right edge = '>' (bit 1).
    Best for shallow trees; depth 6 expands to 64 leaves / 127 lines."""
    depth = len(splits)
    lines: list[str] = []

    def walk(d: int, idx: int, prefix: str, branch: str) -> None:
        if d == depth:
            lines.append(prefix + branch + _leaf_str(leaf_values, leaf_weights, idx, dim, depth))
            return
        node = _split_label(splits[d])
        lines.append(prefix + branch + (node if branch else f"● {node}"))
        if branch == "":
            child_prefix = prefix
        elif branch.startswith("└"):
            child_prefix = prefix + "    "
        else:
            child_prefix = prefix + "│   "
        walk(d + 1, idx, child_prefix, "├─[≤] ")
        walk(d + 1, idx | (1 << d), child_prefix, "└─[>] ")

    walk(0, 0, "", "")
    return "\n".join(lines)


def render_tree_dict(tree: dict[str, Any], full: bool = False) -> str:
    """Convenience: visualize a raw CatBoost JSON oblivious_tree dict directly."""
    splits = tree["splits"]
    values = tree["leaf_values"]
    weights = tree["leaf_weights"]
    dim = max(1, len(values) // len(weights)) if weights else 1
    fn = tree_render if full else tree_repr
    return fn(splits, values, weights, dim)
