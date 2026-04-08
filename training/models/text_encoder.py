# training/models/text_encoder.py
# ── Text Encoder Fine-Tuning Module ──
# 
# Fine-tunes a base MiniLM model on the SciQ dataset using LoRA
# and Contrastive Learning (MultipleNegativesRankingLoss).
# This aligns the text embeddings specifically for the educational science domain.

from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader
from sentence_transformers.evaluation import InformationRetrievalEvaluator
import json

class EducationalEncoder:
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.model = SentenceTransformer(model_name)
        
    def train_contrastive(self, dataset_path: str, output_path: str = "weights/finetuned_minilm", epochs: int = 3):
        """Train the model using MultipleNegativesRankingLoss on contrastive pairs."""
        print(f"[Encoder] Loading training data from {dataset_path}...")
        
        try:
            with open(dataset_path, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[Encoder] Error loading data: {e}. Please run prepare_sciq.py first.")
            return
            
        train_examples = []
        for item in data:
            # Positive pair: (context, correct answer)
            train_examples.append(InputExample(texts=[item["anchor"], item["positive"]]))
            
        # Dataloader
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
        
        # Loss function for contrastive learning
        train_loss = losses.MultipleNegativesRankingLoss(model=self.model)
        
        print(f"[Encoder] Starting fine-tuning for {epochs} epochs...")
        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=100,
            output_path=output_path,
            show_progress_bar=True
        )
        print(f"[Encoder] Fine-tuned model saved to {output_path}")

    def encode(self, texts: list):
        return self.model.encode(texts, convert_to_tensor=True)

if __name__ == "__main__":
    enc = EducationalEncoder()
    print("Encoder initialized successfully.")
