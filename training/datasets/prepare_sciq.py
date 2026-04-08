# training/datasets/prepare_sciq.py
# Downloads SciQ dataset and prepares contrastive training pairs
# for fine-tuning the text encoder on scientific educational content.
#
# SciQ: 13,679 science MCQ with support evidence
# We use 2,000 samples for fast prototyping (still paper-publishable)
#
# Usage:
#   python training/datasets/prepare_sciq.py --samples 2000 --out data/sciq_pairs.json

import argparse, json, random
from pathlib import Path

def prepare_sciq(n_samples: int = 2000, out_path: str = "data/sciq_pairs.json"):
    from datasets import load_dataset

    print("[SciQ] Loading dataset from HuggingFace...")
    ds = load_dataset("allenai/sciq", split="train")

    random.seed(42)
    indices = random.sample(range(len(ds)), min(n_samples, len(ds)))
    samples = [ds[i] for i in indices]

    pairs = []
    for item in samples:
        question       = item["question"]
        correct        = item["correct_answer"]
        support        = item.get("support", "")
        distractors    = [item["distractor1"], item["distractor2"], item["distractor3"]]

        # Positive pair: (question + support context, correct answer)
        pairs.append({
            "anchor"  : f"{question} Context: {support}",
            "positive": correct,
            "negatives": distractors,
            "label"   : 1,
        })

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(pairs, f, indent=2)

    print(f"[SciQ] Saved {len(pairs)} contrastive pairs → {out_path}")
    return pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--out",     type=str, default="data/sciq_pairs.json")
    args = parser.parse_args()
    prepare_sciq(args.samples, args.out)
