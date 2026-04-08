# training/datasets/prepare_misconceptions.py
# Builds a synthetic misconception graph dataset for training T5-small.
#
# Strategy:
#   - For each SciQ item: the CORRECT answer gives us the reference graph.
#   - DISTRACTOR answers are the "wrong student answers" (misconceptions).
#   - We generate (student_answer, reference_text) → JSON_graph pairs.
#
# The T5 model learns: given student's wrong answer + correct reference,
# produce the concept-relation-concept triplets that are MISSING.
#
# Usage:
#   python training/datasets/prepare_misconceptions.py --samples 2000

import argparse, json, re, random
from pathlib import Path

RELATIONS = [
    "IS_A", "HAS_PROPERTY", "USED_FOR", "PART_OF",
    "CAUSES", "REQUIRES", "PRODUCES", "LOCATED_IN", "RELATED_TO"
]

STOP = {"the","a","an","is","of","in","to","and","or","it","this",
        "that","which","with","from","by","on","at","are","was",
        "were","be","been","has","have","had","do","does","can",
        "will","as","for","but","not","its","also"}

def extract_concepts(text: str):
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return list({w for w in words if w not in STOP})[:10]

def make_graph(concepts: list, relations: list = None) -> list:
    """Build a plausible graph from a list of concept words."""
    graph = []
    rels  = relations or RELATIONS
    for i in range(len(concepts) - 1):
        graph.append({
            "s": concepts[i],
            "r": random.choice(rels),
            "o": concepts[i + 1],
        })
    return graph

def prepare_misconceptions(n_samples: int = 2000,
                           out_path: str = "data/misconception_pairs.json"):
    from datasets import load_dataset

    print("[Misconception] Loading SciQ...")
    ds = load_dataset("allenai/sciq", split="train")

    random.seed(42)
    indices = random.sample(range(len(ds)), min(n_samples, len(ds)))

    records = []
    for i in indices:
        item     = ds[i]
        question = item["question"]
        correct  = item["correct_answer"]
        support  = item.get("support", "") or ""
        distractor = random.choice([
            item["distractor1"], item["distractor2"], item["distractor3"]
        ])

        ref_concepts     = extract_concepts(correct + " " + support)
        student_concepts = extract_concepts(distractor)

        ref_graph     = make_graph(ref_concepts)
        student_graph = make_graph(student_concepts)

        # Target: the MISSING triplets (what student should have said)
        missing = [t for t in ref_graph if t not in student_graph]

        if not missing:
            missing = ref_graph[:2]

        input_text = (
            f"Question: {question} "
            f"Student answer: {distractor} "
            f"Reference: {correct}. {support[:200]}"
        )
        target_text = json.dumps(missing)

        records.append({
            "input" : input_text,
            "target": target_text,
            "question": question,
        })

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)

    print(f"[Misconception] Saved {len(records)} pairs → {out_path}")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--out",     type=str, default="data/misconception_pairs.json")
    args = parser.parse_args()
    prepare_misconceptions(args.samples, args.out)
