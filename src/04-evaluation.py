#!/usr/bin/env python3
"""
GNN Model V4 Evaluation Script

This script evaluates the trained GNN V4 model on the held-out test set.
It loads the saved artifacts and computes MAE, RMSE, R² metrics.
Generates diagnostic plots saved to plots/.

Usage:
    python src/04-evaluation.py

Requires artifacts from training (models/gnn_v4/).
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader

from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import joblib

# Project imports
import config
from utils import setup_logger

sns.set_style('whitegrid')


# =============================================================================
# SETUP
# =============================================================================

logger = setup_logger(
    name='gnn_model_v4_eval',
    log_dir=config.LOG_DIR,
    filename='gnn_model_v4_evaluation.log',
    level=logging.INFO,
    mode='w',
)

seed = config.SEED
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device(config.DEVICE)
logger.info('=== STARTING GNN MODEL V4 EVALUATION ===')
logger.info(f"Device: {device}")


# =============================================================================
# LOAD ARTIFACTS
# =============================================================================

artifacts_dir = os.path.join(config.PROJECT_ROOT, 'models', 'gnn_v4')
logger.info(f"Loading artifacts from: {artifacts_dir}")

# Load meta config
meta = joblib.load(os.path.join(artifacts_dir, 'meta.joblib'))
seq_len = meta['seq_len']
context_seq_cols = meta['context_seq_cols']
gru_hidden = meta['gru_hidden']
dropout = meta['dropout']
num_seq_features = len(context_seq_cols)

logger.info(f"Meta config: seq_len={seq_len}, gru_hidden={gru_hidden}, dropout={dropout}")

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


# =============================================================================
# LOAD AND PREPARE TEST DATA
# =============================================================================

logger.info('--- Loading Test Data ---')

clean_path = config.CLEANED_CSV_PATH
df = pd.read_csv(clean_path)

stop_col = 'last_stop_id' if 'last_stop_id' in df.columns else 'stop_id'
df = df.dropna(subset=['timestamp', 'trip_id', stop_col, 'delay_seconds', 'latitude', 'longitude']).copy()
df['dt'] = pd.to_datetime(df['timestamp'], errors='coerce')
df = df.dropna(subset=['dt']).copy()
df = df.sort_values(['trip_id', 'dt']).reset_index(drop=True)

# Feature engineering (same as training)
lag_clip_min = config.GNN_V4_LAG_CLIP_MIN
lag_clip_max = config.GNN_V4_LAG_CLIP_MAX
time_delta_clip = config.GNN_V4_TIME_DELTA_CLIP_SEC
rolling_w = config.GNN_V4_ROLLING_LAG_WINDOW

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

# Apply scalers (transform only, not fit)
df['prev_delay_clipped'] = df['prev_stop_delay'].clip(lag_clip_min, lag_clip_max)
df['rolling_prev_delay_clipped'] = df['rolling_prev_delay'].clip(lag_clip_min, lag_clip_max)

df['prev_delay_scaled'] = scaler_lag.transform(df[['prev_delay_clipped']])
df['rolling_prev_delay_scaled'] = scaler_roll.transform(df[['rolling_prev_delay_clipped']])
df['time_delta_scaled'] = scaler_delta.transform(df[['time_delta_sec']])
df['progress_scaled'] = scaler_prog.transform(df[['progress']])

# Map stop_id to stop_idx
df['stop_idx'] = df[stop_col].astype(str).map(stop_id_to_idx)
# Handle unknown stops (set to -1, will be filtered)
df['stop_idx'] = df['stop_idx'].fillna(-1).astype(np.int64)

# Load test indices
test_idx_path = os.path.join(artifacts_dir, 'test_idx.npy')
if os.path.exists(test_idx_path):
    test_idx = np.load(test_idx_path)
    logger.info(f"Loaded test indices: {len(test_idx):,} samples")
else:
    # Fallback: use 20% of data as test
    from sklearn.model_selection import train_test_split
    indices = np.arange(len(df))
    _, test_idx = train_test_split(indices, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)
    logger.info(f"Generated test indices: {len(test_idx):,} samples")

# Filter out samples with unknown stops
valid_mask = df['stop_idx'].iloc[test_idx] >= 0
test_idx = test_idx[valid_mask.values]
logger.info(f"Valid test samples (known stops): {len(test_idx):,}")


# =============================================================================
# PREPARE TEST DATALOADER
# =============================================================================

X_ctx_base = df[context_seq_cols].to_numpy(dtype=np.float32, copy=False)
stop_idx_base = df['stop_idx'].to_numpy(dtype=np.int64, copy=False)
y_base = df['delay_seconds'].to_numpy(dtype=np.float32, copy=False).reshape(-1, 1)

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
    y = np.empty((batch_size, 1), dtype=np.float32)

    for b, r in enumerate(row_indices):
        pos = int(pos_in_trip[r])
        L = seq_len if pos + 1 >= seq_len else (pos + 1)
        start = r - L + 1
        seq[b, -L:, :] = X_ctx_base[start : r + 1]
        lengths[b] = L
        stop_idx[b] = stop_idx_base[r]
        y[b, 0] = y_base[r, 0]

    return (
        torch.from_numpy(seq),
        torch.from_numpy(lengths),
        torch.from_numpy(stop_idx),
        torch.from_numpy(y),
    )


test_dataset = RowIndexDataset(test_idx)
eval_batch_size = config.GNN_V4_EVAL_BATCH_SIZE
test_loader = DataLoader(
    test_dataset,
    batch_size=eval_batch_size,
    shuffle=False,
    num_workers=config.NUM_WORKERS,
    pin_memory=config.PIN_MEMORY,
    collate_fn=make_window_batch,
)


# =============================================================================
# RUN EVALUATION
# =============================================================================

logger.info('--- Running Evaluation ---')

all_preds = []
all_targets = []

with torch.no_grad():
    for seq_batch, lengths, stop_idx, y_batch in test_loader:
        seq_batch = seq_batch.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        stop_idx = stop_idx.to(device, non_blocking=True)
        out = model(graph_data.x, graph_data.edge_index, seq_batch, lengths, stop_idx)
        all_preds.append(out.detach().cpu().numpy())
        all_targets.append(y_batch.detach().cpu().numpy())

y_pred = np.vstack(all_preds).flatten()
y_true = np.vstack(all_targets).flatten()

# Compute metrics
mae = mean_absolute_error(y_true, y_pred)
rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
r2 = float(r2_score(y_true, y_pred))

logger.info('=== FINAL GNN MODEL V4 RESULTS ===')
logger.info(f"MAE (Mean Absolute Error): {mae:.2f} seconds")
logger.info(f"RMSE (Root Mean Sq Error): {rmse:.2f} seconds")
logger.info(f"R² Score:                  {r2:.4f}")

champion_mae = config.CHAMPION_MAE
logger.info(f"VS Context RF ({champion_mae:.2f}s): {champion_mae - mae:+.2f}s difference")

print("\n" + "="*50)
print("GNN MODEL V4 EVALUATION RESULTS")
print("="*50)
print(f"MAE:  {mae:.2f} seconds")
print(f"RMSE: {rmse:.2f} seconds")
print(f"R²:   {r2:.4f}")
print(f"VS Baseline ({champion_mae:.2f}s): {champion_mae - mae:+.2f}s")
print("="*50 + "\n")


# =============================================================================
# GENERATE DIAGNOSTIC PLOTS
# =============================================================================

logger.info('--- Generating Diagnostic Plots ---')

plots_dir = config.PLOTS_DIR
os.makedirs(plots_dir, exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

plot_n = min(5000, len(y_pred))
plot_indices = np.random.choice(len(y_pred), size=plot_n, replace=False)
y_pred_sub = y_pred[plot_indices]
y_true_sub = y_true[plot_indices]
residuals = y_true_sub - y_pred_sub

# Plot 1: Predicted vs Actual
axes[0].scatter(y_true_sub, y_pred_sub, alpha=0.3, s=10, color='blue')
axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2, label='Perfect Fit')
axes[0].set_title(f"GNN Model V4: Predicted vs Actual\nMAE: {mae:.2f}s | R²: {r2:.2f}")
axes[0].set_xlabel('Actual Delay (s)')
axes[0].set_ylabel('Predicted Delay (s)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Residuals vs Predictions
axes[1].scatter(y_pred_sub, residuals, alpha=0.3, s=10, color='purple')
axes[1].axhline(0, color='red', linestyle='--', lw=2)
axes[1].set_title('Residuals vs Predictions')
axes[1].set_xlabel('Predicted Delay (s)')
axes[1].set_ylabel('Error (s)')
axes[1].grid(True, alpha=0.3)

# Plot 3: Error Distribution
sns.histplot(residuals, bins=50, kde=True, ax=axes[2], color='green')
axes[2].axvline(0, color='red', linestyle='--', lw=2)
axes[2].set_title('Error Distribution')
axes[2].set_xlabel('Error (s)')

fig.tight_layout()
plot_path = os.path.join(plots_dir, 'gnn_model_v4_diagnostics.png')
fig.savefig(plot_path, dpi=150)
plt.close(fig)
logger.info(f"Saved diagnostic plots to: {plot_path}")

# Save metrics to CSV
metrics_df = pd.DataFrame([{
    'model': 'GNN_V4',
    'mae': mae,
    'rmse': rmse,
    'r2': r2,
    'test_samples': len(y_true),
}])
metrics_path = os.path.join(config.DATA_DIR, 'gnn_v4_metrics.csv')
metrics_df.to_csv(metrics_path, index=False)
logger.info(f"Saved metrics to: {metrics_path}")

logger.info('=== GNN MODEL V4 EVALUATION COMPLETE ===')
