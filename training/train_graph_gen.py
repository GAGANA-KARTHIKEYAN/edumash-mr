import os
import json
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys

# Ensure parent dirs are reachable for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.models.misconception_graph_gen import MisconceptionGraphGen, save_graph_gen

class GraphDataset(Dataset):
    def __init__(self, json_path: str):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "input": item["input"],
            "target": item["target"] # The json string of triplets
        }

def train_t5_graph_gen(data_path="data/misconception_pairs.json", epochs=3, batch_size=8, resume=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"========== T5 Graph Generator LoRA Training ({device}) ==========")
    
    if not os.path.exists(data_path):
        print(f"Dataset block missing: {data_path}")
        print("Please run `python training/datasets/prepare_misconceptions.py` first.")
        return

    # 1. Provide the model and wrap it in LoRA
    print("Loading google/mt5-small and applying LoRA PEFT...")
    graph_gen = MisconceptionGraphGen(model_name="google/mt5-small", device=device)
    graph_gen.prepare_for_training(r=8, lora_alpha=16, lora_dropout=0.1)
    
    start_epoch = 0
    checkpoint_dir = Path("weights/checkpoints/t5_graph")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt = checkpoint_dir / "latest_t5.pt"
    
    if resume and latest_ckpt.exists():
        checkpoint = torch.load(latest_ckpt, map_location=device)
        graph_gen.model.load_state_dict(checkpoint['model_state_dict']) # Load LoRA adapters
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}...")

    # 2. Optimizer
    optimizer = torch.optim.AdamW(graph_gen.model.parameters(), lr=3e-4) # Higher LR for LoRA usually
    if resume and latest_ckpt.exists() and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # 3. DataLoader (Train / Val Split)
    dataset = GraphDataset(data_path)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=max(1, batch_size//2), shuffle=False)
    
    print(f"Starting Training: {len(train_dataset)} Train | {len(val_dataset)} Val over {epochs} epochs.")
    
    for epoch in range(start_epoch, epochs):
        graph_gen.model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_dataloader):
            inputs_text = batch["input"]
            targets_text = batch["target"]
            
            optimizer.zero_grad()
            
            inputs = graph_gen.tokenizer(
                inputs_text, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(device)
            
            labels = graph_gen.tokenizer(
                targets_text, return_tensors="pt", padding=True, truncation=True, max_length=256
            ).input_ids.to(device)
            
            labels[labels == graph_gen.tokenizer.pad_token_id] = -100
            
            outputs = graph_gen.model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"[Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(train_dataloader)}] Train Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(train_dataloader)
        
        # --- VALIDATION LOOP ---
        graph_gen.model.eval()
        val_loss = 0
        correct_format = 0
        with torch.no_grad():
            for batch in val_dataloader:
                inputs_text = batch["input"]
                targets_text = batch["target"]
                
                inputs = graph_gen.tokenizer(
                    inputs_text, return_tensors="pt", padding=True, truncation=True, max_length=512
                ).to(device)
                
                labels = graph_gen.tokenizer(
                    targets_text, return_tensors="pt", padding=True, truncation=True, max_length=256
                ).input_ids.to(device)
                labels[labels == graph_gen.tokenizer.pad_token_id] = -100
                
                outputs = graph_gen.model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    labels=labels
                )
                val_loss += outputs.loss.item()
                
                # Check formatting accuracy (did it output Valid JSON Triplets?)
                preds = graph_gen.model.generate(input_ids=inputs.input_ids, max_new_tokens=100)
                pred_texts = graph_gen.tokenizer.batch_decode(preds, skip_special_tokens=True)
                
                for p in pred_texts:
                    try:
                        parsed = json.loads(p)
                        if isinstance(parsed, list) and len(parsed) > 0 and 's' in parsed[0]:
                            correct_format += 1
                    except:
                        pass
        
        avg_val_loss = val_loss / max(1, len(val_dataloader))
        formatting_accuracy = (correct_format / max(1, len(val_dataset))) * 100
        
        print(f"==> Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Graph Accuracy: {formatting_accuracy:.1f}%")
        
        # Checkpoint every epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': graph_gen.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, latest_ckpt)
        
    print("Training finished!")
    save_graph_gen(graph_gen, "weights/t5_graph_gen_final")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--data", type=str, default="data/misconception_pairs.json")
    args = parser.parse_args()
    
    train_t5_graph_gen(data_path=args.data, epochs=args.epochs, resume=args.resume)
