#!/usr/bin/env python3
"""
GNN Model V4 Inference Script

This script runs the trained GNN V4 model on new/unseen data.
It loads the saved artifacts and generates delay predictions.

Usage:
    python src/05-inference.py [--input INPUT_CSV] [--output OUTPUT_CSV]

Defaults:
    --input:  data/vehicle_positions_cleaned.csv
    --output: data/predictions.csv

Requires artifacts from training (models/gnn_v4/).
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader

from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv

import joblib

# Project imports
import config
from utils import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description='GNN V4 Inference')
    parser.add_argument('--input', type=str, default=config.CLEANED_CSV_PATH,
                        help='Path to input CSV file')
    parser.add_argument('--output', type=str, default=os.path.join(config.DATA_DIR, 'predictions.csv'),
                        help='Path to output predictions CSV')
    parser.add_argument('--batch-size', type=int, default=config.GNN_V4_EVAL_BATCH_SIZE,
                        help='Batch size for inference')
    return parser.parse_args()


# =============================================================================
# MODEL DEFINITION (Must match training)
# =============================================================================

class ContextAwareGATv2_GRU_V4(nn.Module):
    def __init__(self, num_node_features: int, num_seq_features: int, gru_hidden: int, dropout: float):
        super().__init__()
        
        self.gat1 = GATv2Conv(num_node_features, 64, heads=4, dropout=dropout)
        self.gat2 = GATv2Conv(64 * 4, 128, heads=1, concat=False, dropout=dropout)
        
        self.gru = nn.GRU(
            input_size=num_seq_features,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )
        
        fusion_dim = 128 + gru_hidden
        self.fc1 = nn.Linear(fusion_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x_nodes, edge_index, seq_batch, lengths, stop_indices):
        x = self.gat1(x_nodes, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        x = F.elu(x)
        stop_emb = x[stop_indices]
        
        lengths_cpu = lengths.detach().to('cpu')
        packed = pack_padded_sequence(
            seq_batch,
            lengths=lengths_cpu,
            batch_first=True,
            enforce_sorted=False,
        )
        _, h_n = self.gru(packed)
        temp_emb = h_n[-1]
        
        fused = torch.cat([stop_emb, temp_emb], dim=1)
        out = self.dropout(F.elu(self.bn1(self.fc1(fused))))
        out = self.dropout(F.elu(self.bn2(self.fc2(out))))
        out = self.dropout(F.elu(self.fc3(out)))
        return self.fc_out(out)


def main():
    args = parse_args()
    
    # Setup logging
    logger = setup_logger(
        name='gnn_model_v4_inference',
        log_dir=config.LOG_DIR,
        filename='run.log',
        level=logging.INFO,
        mode='a',  # Append to run.log
    )
    
    device = torch.device(config.DEVICE)
    logger.info('=== STARTING GNN MODEL V4 INFERENCE ===')
    logger.info(f"Device: {device}")
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")
    
    # ==========================================================================
    # LOAD ARTIFACTS
    # ==========================================================================
    
    artifacts_dir = os.path.join(config.PROJECT_ROOT, 'models', 'gnn_v4')
    logger.info(f"Loading artifacts from: {artifacts_dir}")
    
    # Load meta config
    meta = joblib.load(os.path.join(artifacts_dir, 'meta.joblib'))
    seq_len = meta['seq_len']
    context_seq_cols = meta['context_seq_cols']
    gru_hidden = meta['gru_hidden']
    dropout = meta['dropout']
    num_seq_features = len(context_seq_cols)
    
    # Load graph
    graph_dict = torch.load(os.path.join(artifacts_dir, 'graph.pt'), map_location=device, weights_only=True)
    graph_data = Data(x=graph_dict['x'].to(device), edge_index=graph_dict['edge_index'].to(device))
    
    # Load stop mapping
    stop_id_to_idx = joblib.load(os.path.join(artifacts_dir, 'stop_id_to_idx.joblib'))
    
    # Load scalers
    scalers = joblib.load(os.path.join(artifacts_dir, 'scalers.joblib'))
    scaler_lag = scalers['scaler_lag']
    scaler_roll = scalers['scaler_roll']
    scaler_delta = scalers['scaler_delta']
    scaler_prog = scalers['scaler_prog']
    
    # Load model
    model = ContextAwareGATv2_GRU_V4(
        num_node_features=graph_data.x.shape[1],
        num_seq_features=num_seq_features,
        gru_hidden=gru_hidden,
        dropout=dropout,
    ).to(device)
    
    model_path = os.path.join(artifacts_dir, 'model.pt')
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    logger.info("Model loaded successfully")
    
    # ==========================================================================
    # LOAD AND PREPARE DATA
    # ==========================================================================
    
    logger.info('--- Loading Input Data ---')
    df = pd.read_csv(args.input)
    original_len = len(df)
    logger.info(f"Loaded {original_len:,} rows")
    
    stop_col = 'last_stop_id' if 'last_stop_id' in df.columns else 'stop_id'
    required_cols = ['timestamp', 'trip_id', stop_col, 'latitude', 'longitude']
    
    # Check for required columns
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Keep original index for output mapping
    df['_original_idx'] = df.index
    
    # Basic cleanup
    df = df.dropna(subset=required_cols).copy()
    df['dt'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['dt']).copy()
    df = df.sort_values(['trip_id', 'dt']).reset_index(drop=True)
    logger.info(f"After cleanup: {len(df):,} rows")
    
    # Feature engineering
    lag_clip_min = config.GNN_V4_LAG_CLIP_MIN
    lag_clip_max = config.GNN_V4_LAG_CLIP_MAX
    time_delta_clip = config.GNN_V4_TIME_DELTA_CLIP_SEC
    rolling_w = config.GNN_V4_ROLLING_LAG_WINDOW
    
    # For inference, we may not have actual delay, use 0 as placeholder for lag features
    if 'delay_seconds' not in df.columns:
        df['delay_seconds'] = 0.0
        logger.warning("No 'delay_seconds' column found - using 0 for lag features")
    
    df['prev_stop_delay'] = df.groupby('trip_id')['delay_seconds'].shift(1).fillna(0.0)
    df['rolling_prev_delay'] = (
        df.groupby('trip_id')['prev_stop_delay']
          .rolling(window=rolling_w, min_periods=1)
          .mean()
          .reset_index(level=0, drop=True)
    )
    df['time_delta_sec'] = df.groupby('trip_id')['dt'].diff().dt.total_seconds().fillna(0.0)
    df['time_delta_sec'] = df['time_delta_sec'].clip(0, time_delta_clip)
    
    trip_len = df.groupby('trip_id')[stop_col].transform('size').astype(np.float32)
    stop_sequence = df.groupby('trip_id').cumcount().astype(np.float32)
    df['progress'] = np.where(trip_len > 1, stop_sequence / (trip_len - 1.0), 0.0)
    
    df['hour'] = df['dt'].dt.hour
    df['day_of_week'] = df['dt'].dt.dayofweek
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Apply scalers
    df['prev_delay_clipped'] = df['prev_stop_delay'].clip(lag_clip_min, lag_clip_max)
    df['rolling_prev_delay_clipped'] = df['rolling_prev_delay'].clip(lag_clip_min, lag_clip_max)
    
    df['prev_delay_scaled'] = scaler_lag.transform(df[['prev_delay_clipped']])
    df['rolling_prev_delay_scaled'] = scaler_roll.transform(df[['rolling_prev_delay_clipped']])
    df['time_delta_scaled'] = scaler_delta.transform(df[['time_delta_sec']])
    df['progress_scaled'] = scaler_prog.transform(df[['progress']])
    
    # Map stop_id to stop_idx
    df['stop_idx'] = df[stop_col].astype(str).map(stop_id_to_idx)
    df['stop_idx'] = df['stop_idx'].fillna(-1).astype(np.int64)
    
    # Identify valid rows (known stops)
    valid_mask = df['stop_idx'] >= 0
    valid_indices = np.where(valid_mask)[0]
    logger.info(f"Valid rows (known stops): {len(valid_indices):,}")
    
    if len(valid_indices) == 0:
        logger.error("No valid rows to process - all stops are unknown!")
        sys.exit(1)
    
    # ==========================================================================
    # PREPARE DATALOADER
    # ==========================================================================
    
    X_ctx_base = df[context_seq_cols].to_numpy(dtype=np.float32, copy=False)
    stop_idx_base = df['stop_idx'].to_numpy(dtype=np.int64, copy=False)
    
    n = len(df)
    num_feat = len(context_seq_cols)
    
    # Compute trip positions
    trip_codes, _ = pd.factorize(df['trip_id'].astype(str), sort=False)
    change = np.flatnonzero(np.diff(trip_codes) != 0) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [n]))
    
    pos_in_trip = np.empty((n,), dtype=np.int64)
    for s, e in zip(starts, ends):
        pos_in_trip[s:e] = np.arange(e - s, dtype=np.int64)
    
    class RowIndexDataset(torch.utils.data.Dataset):
        def __init__(self, row_indices: np.ndarray):
            self.row_indices = row_indices
        def __len__(self):
            return int(self.row_indices.shape[0])
        def __getitem__(self, i: int):
            return int(self.row_indices[i])
    
    def make_window_batch(row_indices: list[int]):
        batch_size = len(row_indices)
        seq = np.zeros((batch_size, seq_len, num_feat), dtype=np.float32)
        lengths = np.empty((batch_size,), dtype=np.int64)
        stop_idx = np.empty((batch_size,), dtype=np.int64)
        orig_idx = np.empty((batch_size,), dtype=np.int64)
        
        for b, r in enumerate(row_indices):
            pos = int(pos_in_trip[r])
            L = seq_len if pos + 1 >= seq_len else (pos + 1)
            start = r - L + 1
            seq[b, -L:, :] = X_ctx_base[start : r + 1]
            lengths[b] = L
            stop_idx[b] = stop_idx_base[r]
            orig_idx[b] = r
        
        return (
            torch.from_numpy(seq),
            torch.from_numpy(lengths),
            torch.from_numpy(stop_idx),
            torch.from_numpy(orig_idx),
        )
    
    dataset = RowIndexDataset(valid_indices)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        collate_fn=make_window_batch,
    )
    
    # ==========================================================================
    # RUN INFERENCE
    # ==========================================================================
    
    logger.info('--- Running Inference ---')
    
    all_preds = []
    all_indices = []
    
    with torch.no_grad():
        for seq_batch, lengths, stop_idx, orig_idx in loader:
            seq_batch = seq_batch.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            stop_idx = stop_idx.to(device, non_blocking=True)
            out = model(graph_data.x, graph_data.edge_index, seq_batch, lengths, stop_idx)
            all_preds.append(out.detach().cpu().numpy().flatten())
            all_indices.append(orig_idx.numpy())
    
    predictions = np.concatenate(all_preds)
    row_indices = np.concatenate(all_indices)
    
    # ==========================================================================
    # SAVE OUTPUT
    # ==========================================================================
    
    logger.info('--- Saving Predictions ---')
    
    # Create output dataframe
    df['predicted_delay'] = np.nan
    df.loc[row_indices, 'predicted_delay'] = predictions
    
    # Select output columns
    output_cols = ['_original_idx', 'timestamp', 'trip_id', stop_col, 'latitude', 'longitude']
    if 'delay_seconds' in df.columns:
        output_cols.append('delay_seconds')
    output_cols.append('predicted_delay')
    
    # Rename _original_idx back to original index
    output_df = df[output_cols].copy()
    output_df = output_df.rename(columns={'_original_idx': 'original_row_index'})
    
    # Save to CSV
    output_df.to_csv(args.output, index=False)
    logger.info(f"Saved {len(output_df):,} predictions to: {args.output}")
    
    # Summary statistics
    valid_preds = output_df['predicted_delay'].dropna()
    logger.info(f"Prediction statistics:")
    logger.info(f"  Count: {len(valid_preds):,}")
    logger.info(f"  Mean:  {valid_preds.mean():.2f}s")
    logger.info(f"  Std:   {valid_preds.std():.2f}s")
    logger.info(f"  Min:   {valid_preds.min():.2f}s")
    logger.info(f"  Max:   {valid_preds.max():.2f}s")
    
    print(f"\nPredictions saved to: {args.output}")
    print(f"Valid predictions: {len(valid_preds):,} / {original_len:,} rows")
    
    logger.info('=== GNN MODEL V4 INFERENCE COMPLETE ===')


if __name__ == '__main__':
    main()
