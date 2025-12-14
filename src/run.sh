#!/bin/bash
set -e

echo "================================================"
echo "BKK End-of-Trip Delay Prediction Pipeline"
echo "================================================"
echo "Start time: $(date)"
echo ""

# Ensure we're in the right directory
cd /app/src

# Create directories if they don't exist
mkdir -p /app/log /app/plots /app/models/gnn_v4

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
# Use inference.csv if provided, otherwise use the cleaned training data
if [ -f "/app/data/inference.csv" ]; then
    echo "Found inference.csv - cleaning and running predictions on user-provided data"
    python 02-data-cleanser.py --input /app/data/inference.csv --output /app/data/inference_cleaned.csv --mode inference
    python 05-inference.py --input /app/data/inference_cleaned.csv --output /app/data/predictions.csv
else
    echo "No inference.csv found - running predictions on cleaned training data"
    python 05-inference.py
fi
echo "Inference complete."

echo ""
echo "================================================"
echo "Pipeline finished successfully!"
echo "================================================"
echo "End time: $(date)"
echo ""
echo "Artifacts saved to:"
echo "  - Model: /app/models/gnn_v4/"
echo "  - Plots: /app/plots/"
echo "  - Logs:  /app/log/run.log"
echo "  - Predictions: /app/data/predictions.csv"
