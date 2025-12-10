import argparse
from cats_compiler.compiler.scanner import scan_model_file


def main():
    parser = argparse.ArgumentParser(description="Parse a CatBoost JSON model and print summary.")
    parser.add_argument("model_json", help="Path to CatBoost model saved with format='json'")
    args = parser.parse_args()

    model = scan_model_file(args.model_json)
    print(f"Parsed model: {len(model.trees)} tree(s), scale={model.scale}, bias={model.bias}")
    if model.trees:
        t0 = model.trees[0]
        print(f"First tree: depth={t0.depth}, splits={len(t0.splits)}, leaves={len(t0.leaf_values)}")


if __name__ == "__main__":
    main()
