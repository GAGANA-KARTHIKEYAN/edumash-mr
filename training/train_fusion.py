import os
import json
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
from pathlib import Path
import sys

# Ensure parent dirs are reachable for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.models.fusion_module import MultiModalFusion, save_fusion

class MultimodalDataset(Dataset):
    def __init__(self, json_path: str):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "question": item["question"],
            "image_path": item["image_path"],
            "correct_answer": item["correct_answer"]
        }

def train_fusion_module(data_path="data/scienceqa/multimodal_pairs.json", epochs=5, batch_size=16, resume=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"========== Multimodal Fusion Training ({device}) ==========")
    
    if not os.path.exists(data_path):
        print(f"Dataset block missing: {data_path}")
        print("Please run `python training/datasets/prepare_scienceqa.py` first.")
        return

    # 1. Load Frozen Models (Feature Extractors)
    print("Loading Frozen Feature Extractors (Multilingual MiniLM + CLIP)...")
    text_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device=device)
    
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval() # MUST be frozen!

    # 2. Load Trainable Fusion Module
    print("Loading Trainable Cross-Attention Fusion Layer...")
    fusion_model = MultiModalFusion(unified_dim=384).to(device)
    
    start_epoch = 0
    checkpoint_dir = Path("weights/checkpoints/fusion")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    latest_ckpt = checkpoint_dir / "latest_fusion.pt"
    if resume and latest_ckpt.exists():
        checkpoint = torch.load(latest_ckpt, map_location=device)
        fusion_model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}...")
        
    # 3. Predictor Head (Target: Classify the correct answer embedding)
    # We map fused representation back to text space to compare with sentence embedding of actual answer.
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(fusion_model.parameters(), lr=1e-4)
    
    if resume and latest_ckpt.exists() and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 4. Prepare DataLoader
    dataset = MultimodalDataset(data_path)
    # small batch so CPU doesn't die
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Starting Training: {len(dataset)} items over {epochs} epochs.")
    from PIL import Image

    for epoch in range(start_epoch, epochs):
        fusion_model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            questions = batch["question"]
            images = batch["image_path"]
            answers = batch["correct_answer"]
            
            optimizer.zero_grad()
            
            # --- Extract Text Features (Frozen) ---
            with torch.no_grad():
                text_embs = text_model.encode(questions, convert_to_tensor=True).clone().to(device)
                target_embs = text_model.encode(answers, convert_to_tensor=True).clone().to(device)
            
            # --- Extract Image Features (Frozen) ---
            valid_image_embs = []
            for img_path in images:
                try:
                    img = Image.open(img_path).convert("RGB")
                    inputs = clip_processor(images=img, return_tensors="pt").to(device)
                    with torch.no_grad():
                        vision_outputs = clip_model.vision_model(**inputs)
                        pooled = vision_outputs[1]  # (1, 768)
                        projected = clip_model.visual_projection(pooled) # (1, 512)
                        img_feature = projected[0]
                    valid_image_embs.append(img_feature)
                except Exception as e:
                    # Fallback on zero-vector if image corrupted
                    print(f"Warning corrupted image {img_path}: {e}")
                    valid_image_embs.append(torch.zeros(512, device=device))
                    
            image_embs = torch.stack(valid_image_embs)
            
            # --- Train Fusion Module ---
            fused_representation = fusion_model(text_emb=text_embs, image_emb=image_embs)
            
            # We want the Fused Representation to map closely to the Answer Embedding
            loss = loss_fn(fused_representation, target_embs)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"[Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(dataloader)}] MSE Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(dataloader)
        print(f"==> Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}")
        
        # Checkpoint every epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': fusion_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, latest_ckpt)
        
    print("Training finished!")
    save_fusion(fusion_model, "weights/fusion_module_final.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    
    train_fusion_module(epochs=args.epochs, resume=args.resume)
