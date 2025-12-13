#!/bin/bash
set -e

echo "================================================"
echo "BKK End-of-Trip Delay Prediction Pipeline"
echo "================================================"

cd /app/src

echo ""
echo "[Step 1/4] Data Preprocessing..."
echo "------------------------------------------------"
python 02-data-cleanser.py
echo "Data preprocessing complete."

echo ""
echo "[Step 2/4] Model Training (GNN V4)..."
echo "------------------------------------------------"
python 03-training.py
echo "Training complete."

echo ""
echo "[Step 3/4] Model Evaluation..."
echo "------------------------------------------------"
python 04-evaluation.py
echo "Evaluation complete."

echo ""
echo "[Step 4/4] Running Inference..."
echo "------------------------------------------------"
python 05-inference.py
echo "Inference complete."

echo ""
echo "================================================"
echo "Pipeline finished successfully!"
echo "================================================"
echo ""
echo "Artifacts saved to:"
echo "  - Model: /app/models/gnn_v4/"
echo "  - Plots: /app/plots/"
echo "  - Logs:  /app/log/"
echo "  - Predictions: /app/data/predictions.csv"
