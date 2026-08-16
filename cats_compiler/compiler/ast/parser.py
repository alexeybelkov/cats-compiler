from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any

from cats_compiler.compiler.ast.tree import Tree


@dataclass
class ModelFile:
    model_file: dict[str, Any]
    @property
    def trees(self) -> list[Tree]:
        return [Tree.from_dict(tree) for tree in self.model_file['oblivious_trees']]


def parse_model_file_from_json(file_path: str | Path) -> ModelFile:
    if isinstance(file_path, str): file_path = Path(file_path)
    with file_path.open('r', encoding='utf-8') as f: model_file = json.load(f)
    return ModelFile(model_file)
