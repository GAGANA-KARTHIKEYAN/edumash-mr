# training/models/misconception_graph_gen.py
# ── NOVEL COMPONENT #2: Misconception Graph Generator ──
#
# A T5-small model fine-tuned to extract Concept Knowledge Graphs (as JSON) from text.
# Crucially, when given BOTH a Reference Text and a Student's (wrong) Answer,
# it generates a graph of the MISSING or INCORRECT concepts/relations.
#
# Base model: flan-t5-small or t5-small
# Fine-tuning uses LoRA for efficiency.

import torch
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
try:
    from peft import LoraConfig, get_peft_model, TaskType
except ImportError:
    pass

class MisconceptionGraphGen:
    def __init__(self, model_name: str = "google/mt5-small", device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.is_lora = False

    def prepare_for_training(self, r=8, lora_alpha=16, lora_dropout=0.1):
        """Prepare model for parameter-efficient fine-tuning with LoRA."""
        config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q", "v"] # standard T5 targets
        )
        self.model = get_peft_model(self.model, config)
        self.model.print_trainable_parameters()
        self.is_lora = True

    def generate_graph(self, question: str, student_answer: str, reference: str) -> list:
        """
        Inference method to extract missing concept triplets.
        Input format: "Question: Q Student answer: A Reference: R"
        Output: JSON list of triplets: [{"s": subject, "r": relation, "o": object}]
        """
        prompt = f"Question: {question} Student answer: {student_answer} Reference: {reference}"
        
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generation with constrained parameters to encourage JSON-like structure
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.3,
                num_beams=3,
                forced_eos_token_id=self.tokenizer.eos_token_id
            )
            
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Try to parse the result as JSON (since it's trained to output JSON)
        try:
            # Simple cleanup in case model generated extra text
            start = decoded.find("[")
            end = decoded.rfind("]") + 1
            if start != -1 and end != 0:
                json_str = decoded[start:end]
                return json.loads(json_str)
            else:
                print(f"[GraphGen] No JSON brackets found in output: {decoded}")
                return []
        except json.JSONDecodeError:
            print(f"[GraphGen] Failed to parse JSON graph: {decoded}")
            return []

def save_graph_gen(mg_gen: MisconceptionGraphGen, path: str = "weights/t5_graph_gen"):
    from pathlib import Path
    Path(path).mkdir(parents=True, exist_ok=True)
    if mg_gen.is_lora:
        mg_gen.model.save_pretrained(path)
    else:
        mg_gen.model.save_pretrained(path)
    mg_gen.tokenizer.save_pretrained(path)
    print(f"[GraphGen] Saved to {path}")

def load_graph_gen(path: str = "weights/t5_graph_gen", base_model: str = "google/mt5-small") -> MisconceptionGraphGen:
    from peft import PeftModel, PeftConfig
    import os
    
    gen = MisconceptionGraphGen(model_name=base_model)
    
    if os.path.exists(path):
        # Attempt to load as LoRA adapter
        try:
            gen.model = PeftModel.from_pretrained(gen.model, path)
            print(f"[GraphGen] Loaded fine-tuned LoRA weights from {path}")
        except Exception as e:
            print(f"[GraphGen] Could not load as LoRA, might be full model or untrained: {e}")
    else:
        print(f"[GraphGen] Weights not found at {path}, using base untrained model.")
        
    return gen

if __name__ == "__main__":
    print("Testing Graph Generator Initialization...")
    mg = MisconceptionGraphGen()
    triplets = mg.generate_graph(
        "What do plants need for photosynthesis?",
        "Plants need oxygen to make food.",
        "Photosynthesis uses sunlight, water, and carbon dioxide."
    )
    print(triplets)    
