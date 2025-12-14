#!/usr/bin/env python3
"""
GNN Model V4 Training Script (GATv2Conv + GRU)

This script trains the GNN V4 model for transit delay prediction.
It loads cleaned vehicle position data, constructs a stop graph,
and trains a temporal-spatial model using GATv2Conv + GRU.

Usage:
    python src/03-training.py

All hyperparameters are configured via src/config.py (GNN_V4_* keys).
Artifacts are saved to models/gnn_v4/ for inference.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

import joblib

# Project imports
import config
from utils import setup_logger


# =============================================================================
# STEP 0: RUNTIME SETUP
# =============================================================================

logger = setup_logger(
    name='gnn_model_v4',
    log_dir=config.LOG_DIR,
    filename='run.log',
    level=logging.INFO,
    mode='a',  # Append to run.log
)

seed = config.SEED
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device(config.DEVICE)
logger.info('=== STARTING GNN MODEL V4 TRAINING ===')
logger.info(f"Device: {device}")
if torch.cuda.is_available():
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")


# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================

logger.info('--- Loading Data ---')
clean_path = config.CLEANED_CSV_PATH
logger.info(f"Loading data from: {clean_path}")
df = pd.read_csv(clean_path)

required_cols = ['timestamp', 'trip_id', 'delay_seconds', 'latitude', 'longitude']
missing_required = [c for c in required_cols if c not in df.columns]
if missing_required:
    raise ValueError(f"Missing required columns: {missing_required}")

# Choose stop column name consistently
stop_col = 'last_stop_id' if 'last_stop_id' in df.columns else 'stop_id'
if stop_col not in df.columns:
    raise ValueError("No stop column found (expected 'last_stop_id' or 'stop_id')")

# Basic cleanup
df = df.dropna(subset=['timestamp', 'trip_id', stop_col, 'delay_seconds', 'latitude', 'longitude']).copy()
df['dt'] = pd.to_datetime(df['timestamp'], errors='coerce')
df = df.dropna(subset=['dt']).copy()

# Canonical sort for temporal feature engineering
df = df.sort_values(['trip_id', 'dt']).reset_index(drop=True)
logger.info(f"Rows after basic cleanup: {len(df):,}")


# =============================================================================
# STEP 2: FEATURE ENGINEERING
# =============================================================================

logger.info('--- Generating V4 Features ---')

# Config-driven knobs
lag_clip_min = config.GNN_V4_LAG_CLIP_MIN
lag_clip_max = config.GNN_V4_LAG_CLIP_MAX
time_delta_clip = config.GNN_V4_TIME_DELTA_CLIP_SEC
rolling_w = config.GNN_V4_ROLLING_LAG_WINDOW

# 1) Lag feature
df['prev_stop_delay'] = df.groupby('trip_id')['delay_seconds'].shift(1).fillna(0.0)

# 2) Rolling lag mean
df['rolling_prev_delay'] = (
    df.groupby('trip_id')['prev_stop_delay']
      .rolling(window=rolling_w, min_periods=1)
      .mean()
      .reset_index(level=0, drop=True)
)

# 3) Time delta between consecutive observations within trip
df['time_delta_sec'] = df.groupby('trip_id')['dt'].diff().dt.total_seconds().fillna(0.0)
df['time_delta_sec'] = df['time_delta_sec'].clip(0, time_delta_clip)

# 4) Progress in trip (normalized)
trip_len = df.groupby('trip_id')[stop_col].transform('size').astype(np.float32)
stop_sequence = df.groupby('trip_id').cumcount().astype(np.float32)
df['progress'] = np.where(trip_len > 1, stop_sequence / (trip_len - 1.0), 0.0)

# 5) Cyclical time embeddings
df['hour'] = df['dt'].dt.hour
df['day_of_week'] = df['dt'].dt.dayofweek
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

# 6) Stop-level historical mean delay (static signal)
global_mean = float(df['delay_seconds'].mean())
stop_history = df.groupby(stop_col)['delay_seconds'].mean()
df['history_mean'] = df[stop_col].map(stop_history).fillna(global_mean)

# Scaling
logger.info('Scaling features...')

# A) Node features (static): lat/lon + history
scaler_nodes = StandardScaler()
df[['lat_scaled', 'lon_scaled', 'hist_scaled']] = scaler_nodes.fit_transform(
    df[['latitude', 'longitude', 'history_mean']]
)

# B) Dynamic features (context stream)
df['prev_delay_clipped'] = df['prev_stop_delay'].clip(lag_clip_min, lag_clip_max)
df['rolling_prev_delay_clipped'] = df['rolling_prev_delay'].clip(lag_clip_min, lag_clip_max)

scaler_lag = StandardScaler()
df['prev_delay_scaled'] = scaler_lag.fit_transform(df[['prev_delay_clipped']])

scaler_roll = StandardScaler()
df['rolling_prev_delay_scaled'] = scaler_roll.fit_transform(df[['rolling_prev_delay_clipped']])

scaler_delta = StandardScaler()
df['time_delta_scaled'] = scaler_delta.fit_transform(df[['time_delta_sec']])

scaler_prog = StandardScaler()
df['progress_scaled'] = scaler_prog.fit_transform(df[['progress']])

logger.info('Feature engineering complete.')


# =============================================================================
# STEP 3: GRAPH CONSTRUCTION
# =============================================================================

logger.info('--- Constructing Transit Graph ---')

stop_encoder = LabelEncoder()
df['stop_idx'] = stop_encoder.fit_transform(df[stop_col].astype(str))

# Node feature tensor
node_cols = ['lat_scaled', 'lon_scaled', 'hist_scaled']
node_features_df = df.groupby('stop_idx')[node_cols].mean()
x = torch.tensor(node_features_df.values, dtype=torch.float32)

# Edges from sequential transitions within trip
df_sorted = df.sort_values(by=['trip_id', 'dt']).copy()
df_sorted['next_stop_idx'] = df_sorted.groupby('trip_id')['stop_idx'].shift(-1)
edges_df = df_sorted.dropna(subset=['next_stop_idx'])
unique_edges = edges_df[['stop_idx', 'next_stop_idx']].drop_duplicates()
edge_index = torch.tensor(unique_edges.values.T, dtype=torch.long)

graph_data = Data(x=x, edge_index=edge_index).to(device)
logger.info(f"Graph ready | nodes={graph_data.num_nodes:,} edges={graph_data.num_edges:,}")


# =============================================================================
# STEP 4: WINDOWED SEQUENCE DATASET
# =============================================================================

logger.info('--- Preparing temporal windows for GRU ---')

seq_len = config.GNN_V4_SEQ_LEN
split_by_trip = config.GNN_V4_SPLIT_BY_TRIP

# Dynamic context features for temporal sequence
context_seq_cols = [
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
    'prev_delay_scaled', 'rolling_prev_delay_scaled',
    'time_delta_scaled', 'progress_scaled',
]

X_ctx_base = df[context_seq_cols].to_numpy(dtype=np.float32, copy=False)
stop_idx_base = df['stop_idx'].to_numpy(dtype=np.int64, copy=False)
y_base = df['delay_seconds'].to_numpy(dtype=np.float32, copy=False).reshape(-1, 1)

n = len(df)
num_feat = len(context_seq_cols)
logger.info(f"Base arrays | rows={n:,} seq_len={seq_len} features={num_feat}")

# Precompute trip segment info
trip_codes, _ = pd.factorize(df['trip_id'].astype(str), sort=False)
change = np.flatnonzero(np.diff(trip_codes) != 0) + 1
starts = np.concatenate(([0], change))
ends = np.concatenate((change, [n]))

trip_start = np.empty((n,), dtype=np.int64)
pos_in_trip = np.empty((n,), dtype=np.int64)
for s, e in zip(starts, ends):
    trip_start[s:e] = s
    pos_in_trip[s:e] = np.arange(e - s, dtype=np.int64)

# Train / Val / Test split
random_state = config.RANDOM_STATE
test_size = config.TEST_SIZE
val_size = config.VAL_SIZE
indices = np.arange(n)

if split_by_trip:
    trips = df['trip_id'].astype(str).unique()
    trainval_trips, test_trips = train_test_split(trips, test_size=test_size, random_state=random_state)
    val_rel = val_size / max(1e-12, (1.0 - test_size))
    val_rel = min(max(val_rel, 0.0), 0.5)
    train_trips, val_trips = train_test_split(trainval_trips, test_size=val_rel, random_state=random_state)

    train_mask = df['trip_id'].astype(str).isin(train_trips).to_numpy()
    val_mask = df['trip_id'].astype(str).isin(val_trips).to_numpy()
    test_mask = df['trip_id'].astype(str).isin(test_trips).to_numpy()

    train_idx = indices[train_mask]
    val_idx = indices[val_mask]
    test_idx = indices[test_mask]
else:
    trainval_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_state)
    val_rel = val_size / max(1e-12, (1.0 - test_size))
    val_rel = min(max(val_rel, 0.0), 0.5)
    train_idx, val_idx = train_test_split(trainval_idx, test_size=val_rel, random_state=random_state)

logger.info(f"Split sizes | train={len(train_idx):,} val={len(val_idx):,} test={len(test_idx):,}")


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


train_dataset = RowIndexDataset(train_idx)
val_dataset = RowIndexDataset(val_idx)
test_dataset = RowIndexDataset(test_idx)


# =============================================================================
# STEP 5: MODEL DEFINITION
# =============================================================================

logger.info('--- Defining Model V4 (GATv2Conv + GRU) ---')

dropout = config.GNN_V4_DROPOUT
gru_hidden = config.GNN_V4_GRU_HIDDEN
num_seq_features = len(context_seq_cols)


class ContextAwareGATv2_GRU_V4(nn.Module):
    def __init__(self, num_node_features: int, num_seq_features: int, gru_hidden: int, dropout: float):
        super().__init__()
        
        # Spatial encoder
        self.gat1 = GATv2Conv(num_node_features, 64, heads=4, dropout=dropout)
        self.gat2 = GATv2Conv(64 * 4, 128, heads=1, concat=False, dropout=dropout)
        
        # Temporal encoder
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
        # Spatial
        x = self.gat1(x_nodes, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        x = F.elu(x)
        stop_emb = x[stop_indices]
        
        # Temporal
        lengths_cpu = lengths.detach().to('cpu')
        packed = pack_padded_sequence(
            seq_batch,
            lengths=lengths_cpu,
            batch_first=True,
            enforce_sorted=False,
        )
        _, h_n = self.gru(packed)
        temp_emb = h_n[-1]
        
        # Fuse + regress
        fused = torch.cat([stop_emb, temp_emb], dim=1)
        out = self.dropout(F.elu(self.bn1(self.fc1(fused))))
        out = self.dropout(F.elu(self.bn2(self.fc2(out))))
        out = self.dropout(F.elu(self.fc3(out)))
        return self.fc_out(out)


model = ContextAwareGATv2_GRU_V4(
    num_node_features=graph_data.x.shape[1],
    num_seq_features=num_seq_features,
    gru_hidden=gru_hidden,
    dropout=dropout,
).to(device)

# Log model architecture summary
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
total_params = trainable_params + non_trainable_params

logger.info('=== MODEL ARCHITECTURE ===')
logger.info(f"Model: ContextAwareGATv2_GRU_V4")
logger.info(f"  Spatial encoder: GATv2Conv(in={graph_data.x.shape[1]}) -> 64*4 -> 128")
logger.info(f"  Temporal encoder: GRU(in={num_seq_features}, hidden={gru_hidden})")
logger.info(f"  Fusion MLP: {128 + gru_hidden} -> 256 -> 128 -> 64 -> 1")
logger.info(f"  Dropout: {dropout}")
logger.info(f"Parameters:")
logger.info(f"  Trainable:     {trainable_params:,}")
logger.info(f"  Non-trainable: {non_trainable_params:,}")
logger.info(f"  Total:         {total_params:,}")
logger.info('=' * 30)


# =============================================================================
# STEP 6: TRAINING
# =============================================================================

logger.info('--- Preparing DataLoaders + Training ---')

batch_size = config.GNN_V4_BATCH_SIZE
num_epochs = config.GNN_V4_NUM_EPOCHS
learning_rate = config.GNN_V4_LR
weight_decay = config.GNN_V4_WEIGHT_DECAY
num_workers = config.NUM_WORKERS
pin_memory = config.PIN_MEMORY

sched_factor = config.GNN_V4_SCHED_FACTOR
sched_patience = config.GNN_V4_SCHED_PATIENCE
sched_threshold = config.GNN_V4_SCHED_THRESHOLD
sched_min_lr = config.GNN_V4_SCHED_MIN_LR
sched_cooldown = config.GNN_V4_SCHED_COOLDOWN
champion_mae = config.CHAMPION_MAE

logger.info(
    f"Hyperparams | epochs={num_epochs} batch_size={batch_size} lr={learning_rate} "
    f"seq_len={seq_len} gru_hidden={gru_hidden} dropout={dropout}"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory,
    drop_last=False,
    collate_fn=make_window_batch,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
    drop_last=False,
    collate_fn=make_window_batch,
)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=sched_factor,
    patience=sched_patience,
    threshold=sched_threshold,
    cooldown=sched_cooldown,
    min_lr=sched_min_lr,
)
criterion = nn.L1Loss()


@torch.no_grad()
def evaluate_mae(loader) -> float:
    model.eval()
    total = 0.0
    n_batches = 0
    for seq_batch, lengths, stop_idx, y_batch in loader:
        seq_batch = seq_batch.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        stop_idx = stop_idx.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)
        out = model(graph_data.x, graph_data.edge_index, seq_batch, lengths, stop_idx)
        loss = criterion(out, y_batch)
        total += float(loss.item())
        n_batches += 1
    return total / max(1, n_batches)


logger.info('Starting training...')
best_val = float('inf')
history = {'train_mae': [], 'val_mae': [], 'lr': []}

for epoch in range(num_epochs):
    model.train()
    total_train = 0.0
    n_train_batches = 0
    for seq_batch, lengths, stop_idx, y_batch in train_loader:
        seq_batch = seq_batch.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        stop_idx = stop_idx.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        out = model(graph_data.x, graph_data.edge_index, seq_batch, lengths, stop_idx)
        loss = criterion(out, y_batch)
        loss.backward()
        optimizer.step()

        total_train += float(loss.item())
        n_train_batches += 1

    train_mae = total_train / max(1, n_train_batches)
    val_mae = evaluate_mae(val_loader)

    lr_before = float(optimizer.param_groups[0]['lr'])
    scheduler.step(val_mae)
    lr_after = float(optimizer.param_groups[0]['lr'])
    lr_note = 'LR↓' if lr_after < lr_before else 'LR='
    
    if val_mae < (best_val - sched_threshold):
        best_val = val_mae

    history['train_mae'].append(train_mae)
    history['val_mae'].append(val_mae)
    history['lr'].append(lr_after)

    delta_to_champion = val_mae - champion_mae
    logger.info(
        f"Epoch {epoch+1:03d}/{num_epochs} | "
        f"train_MAE={train_mae:.2f}s | val_MAE={val_mae:.2f}s | best_val={best_val:.2f}s | "
        f"Δ_vs_{champion_mae:.2f}s={delta_to_champion:+.2f}s | {lr_note} {lr_after:.6g}"
    )

logger.info('Training complete.')


# =============================================================================
# STEP 7: EXPORT ARTIFACTS
# =============================================================================

logger.info('--- Exporting V4 Artifacts ---')

artifacts_dir = os.path.join(config.PROJECT_ROOT, 'models', 'gnn_v4')
os.makedirs(artifacts_dir, exist_ok=True)

# 1) Model weights
model_path = os.path.join(artifacts_dir, 'model.pt')
torch.save(model.state_dict(), model_path)
logger.info(f"Saved model weights to: {model_path}")

# 2) Graph tensors (CPU)
graph_path = os.path.join(artifacts_dir, 'graph.pt')
torch.save({'x': graph_data.x.detach().cpu(), 'edge_index': graph_data.edge_index.detach().cpu()}, graph_path)
logger.info(f"Saved graph to: {graph_path}")

# 3) Stop mapping (string stop_id -> int stop_idx)
stop_id_to_idx = {str(stop_id): int(idx) for idx, stop_id in enumerate(stop_encoder.classes_)}
joblib.dump(stop_id_to_idx, os.path.join(artifacts_dir, 'stop_id_to_idx.joblib'))
logger.info(f"Saved stop mapping")

# 4) Scalers used by the temporal features
joblib.dump({
    'scaler_lag': scaler_lag,
    'scaler_roll': scaler_roll,
    'scaler_delta': scaler_delta,
    'scaler_prog': scaler_prog,
}, os.path.join(artifacts_dir, 'scalers.joblib'))
logger.info(f"Saved scalers")

# 5) Meta config required to rebuild the exact architecture + feature order
joblib.dump({
    'seq_len': int(seq_len),
    'context_seq_cols': list(context_seq_cols),
    'gru_hidden': int(gru_hidden),
    'dropout': float(dropout),
}, os.path.join(artifacts_dir, 'meta.joblib'))
logger.info(f"Saved meta config")

# 6) Save test indices for evaluation script
np.save(os.path.join(artifacts_dir, 'test_idx.npy'), test_idx)
logger.info(f"Saved test indices")

logger.info(f"All V4 artifacts exported to: {artifacts_dir}")
logger.info('=== GNN MODEL V4 TRAINING COMPLETE ===')
