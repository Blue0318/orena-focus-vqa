"""
Run inference for a given track.

Usage:
    python scripts/run_inference.py --config configs/frame.yaml
"""
import argparse
import yaml

from focus_vqa.data.loader import load_dataset, get_requests_and_references
from focus_vqa.models.qwen_vl import QwenVLModel
from focus_vqa.evaluation.metrics import run_evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to track YAML config")
    parser.add_argument("--limit", type=int, default=None, help="Limit to N samples (for debugging)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print(f"Track: {cfg['track']} | Model: {cfg['model']['name']}")

    # Load data
    ds = load_dataset(cfg)
    requests, references = get_requests_and_references(ds)

    if args.limit:
        requests = requests[:args.limit]
        references = references[:args.limit]

    print(f"Loaded {len(requests)} samples")

    # Load model
    model = QwenVLModel(cfg)
    model.load()

    # Run inference
    print("Running inference...")
    answers = model.predict_batch(requests)

    # Evaluate
    output_dir = cfg.get("logging", {}).get("output_dir", "results")
    run_evaluation(requests, references, answers, output_dir=output_dir)


if __name__ == "__main__":
    main()
