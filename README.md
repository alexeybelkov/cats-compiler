# Cats Compiler
LLVM-based compiler for CatBoost trees

This repository aims to provide an LLVM frontend for CatBoost models. As a first step, it includes a Python parser that can read CatBoost JSON model dumps and produce a simple intermediate representation (IR) of the model's trees.

Usage
- Export your CatBoost model to JSON:
  - Python example:
    - model.save_model("model.json", format="json")
- Parse and print a summary via CLI:
  - python -m cats_compiler.main model.json
- Parse programmatically:
  - from cats_compiler.compiler.scanner import scan_model_file
  - model = scan_model_file("model.json")
  - print(len(model.trees), model.scale, model.bias)

Notes
- Currently supports CatBoost "oblivious_trees" with FloatFeature and OneHotFeature splits.
- The IR lives in cats_compiler/compiler/parser.py (Model, Tree, Split, Feature).
- This IR can be used to generate LLVM IR in follow-up steps.
