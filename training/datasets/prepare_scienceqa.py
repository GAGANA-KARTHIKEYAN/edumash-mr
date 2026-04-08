# training/datasets/prepare_scienceqa.py
# Downloads ScienceQA dataset and prepares it for MultiModal Fusion training.
#
# ScienceQA: Multi-modal QA dataset containing science questions with
# accompanying diagrams/images.
# We use this specifically to teach our Cross-Attention Fusion layer how to
# blend text embeddings and visual embeddings.
#
# Usage:
#   python training/datasets/prepare_scienceqa.py --samples 1500

import argparse, json, random, os
from pathlib import Path

def prepare_scienceqa(n_samples: int = 1500, out_dir: str = "data/scienceqa/"):
    from datasets import load_dataset
    
    print("[ScienceQA] Loading multimodal dataset from HuggingFace...")
    # Loading the train split of ScienceQA
    try:
        ds = load_dataset("derek-thomas/ScienceQA", split="train")
    except Exception as e:
        print(f"[ScienceQA] Error loading: {e}. Trying raw repo...")
        # fallback if name changed or something
        ds = load_dataset("science_qa", split="train")
    
    # Filter for samples that actually have an image (multimodal context)
    multimodal_samples = [item for item in ds if item.get("image") is not None]
    print(f"[ScienceQA] Found {len(multimodal_samples)} samples with images.")
    
    random.seed(42)
    indices = random.sample(range(len(multimodal_samples)), min(n_samples, len(multimodal_samples)))
    
    records = []
    
    # Create directories
    image_dir = os.path.join(out_dir, "images")
    Path(image_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"[ScienceQA] Processing {len(indices)} samples and saving images...")
    for idx_num, i in enumerate(indices):
        item = multimodal_samples[i]
        
        question = item["question"]
        choices = item["choices"]
        answer_idx = item["answer"]
        hint = item.get("hint", "")
        
        correct_answer = choices[answer_idx]
        
        # Save image locally
        img = item["image"]
        img_name = f"sqa_{idx_num}.jpg"
        img_path = os.path.join(image_dir, img_name)
        
        # Convert to RGB and save (to avoid PNG RGBA issues with CLIP)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(img_path)
        
        # We need the fusion layer to predict the answer from (question text + image context)
        records.append({
            "id": f"sqa_{idx_num}",
            "question": question,
            "hint": hint,
            "image_path": img_path,
            "correct_answer": correct_answer
        })
        
    json_path = os.path.join(out_dir, "multimodal_pairs.json")
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)
        
    print(f"[ScienceQA] Saved {len(records)} multimodal pairs → {json_path}")
    print(f"[ScienceQA] Saved images to → {image_dir}")
    return records

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1500)
    parser.add_argument("--out_dir", type=str, default="data/scienceqa/")
    args = parser.parse_args()
    prepare_scienceqa(args.samples, args.out_dir)
