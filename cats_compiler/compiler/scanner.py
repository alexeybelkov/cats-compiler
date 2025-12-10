from pathlib import Path
from dataclasses import dataclass
from typing import Iterable


def scan_tree(lines: Iterable[str]) -> dict:
  pass

def scan_model_file(file_path: str | Path):
  if isinstance(file_path, str): file_path = Path(file_path)
  

import json as _json
from .parser import Model as _Model


def scan_tree(lines: Iterable[str]) -> dict:  # type: ignore[override]
  raise NotImplementedError("scan_tree is not implemented; use scan_model_file for JSON models.")


def scan_model_file(file_path: str|Path):  # type: ignore[override]
  if isinstance(file_path, str): file_path = Path(file_path)
  with file_path.open("r", encoding="utf-8") as _f: _data = _json.load(_f)
  return _Model.from_catboost_json(_data)
