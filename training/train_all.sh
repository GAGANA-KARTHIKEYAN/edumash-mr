#!/bin/bash
# training/train_all.sh
# 
# Main pipeline script to execute all ML dataset prep and partial training.
# Recommendation: Run this on Google Colab for the T5 Graph Generator phase.

echo "=========================================================="
echo "🚀 eduMASH-MR Full ML Training Pipeline"
echo "=========================================================="

echo "[1/5] Preparing SciQ Contrastive Dataset..."
python training/datasets/prepare_sciq.py --samples 2000 --out data/sciq_pairs.json

echo ""
echo "[2/5] Preparing Misconception Graph Dataset..."
python training/datasets/prepare_misconceptions.py --samples 2000 --out data/misconception_pairs.json

echo ""
echo "[3/5] Preparing Multi-Modal ScienceQA Dataset (Text + Image)..."
python training/datasets/prepare_scienceqa.py --samples 1500

echo ""
echo "[4/5] Fine-tuning Educational Text Encoder (MiniLM)..."
# In a real run, you would execute an actual train_encoder.py here.
# For now, we test initialization:
python training/models/text_encoder.py

echo ""
echo "[5/5] Cross-Modal Fusion & GNN Modality Testing..."
python training/models/fusion_module.py
python inference/graph_comparator.py
python training/models/misconception_graph_gen.py

echo ""
echo "=========================================================="
echo "✅ Pipeline setup complete."
echo "NOTE: Full T5-small and Fusion module training requires GPU."
echo "Please zip this 'training' folder and upload to Google Colab for fast execution."
echo "=========================================================="
