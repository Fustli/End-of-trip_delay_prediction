from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from torch_geometric.nn import GATv2Conv


@dataclass(frozen=True)
class V4Artifacts:
    seq_len: int
    context_seq_cols: list[str]
    stop_id_to_idx: dict[str, int]
    graph_x: torch.Tensor
    graph_edge_index: torch.Tensor
    scaler_lag: object
    scaler_roll: object
    scaler_delta: object
    scaler_prog: object


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

        lengths_cpu = lengths.detach().to("cpu")
        packed = pack_padded_sequence(seq_batch, lengths=lengths_cpu, batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)
        temp_emb = h_n[-1]

        fused = torch.cat([stop_emb, temp_emb], dim=1)
        out = self.dropout(F.elu(self.bn1(self.fc1(fused))))
        out = self.dropout(F.elu(self.bn2(self.fc2(out))))
        out = self.dropout(F.elu(self.fc3(out)))
        return self.fc_out(out)


def _safe_float(x: float) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _scale_1d(scaler, value: float) -> float:
    arr = np.array([[value]], dtype=np.float32)
    out = scaler.transform(arr)
    return float(out[0, 0])


class V4Predictor:
    def __init__(self, artifacts_dir: str, device: Optional[str] = None):
        self.artifacts_dir = artifacts_dir
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self._artifacts: Optional[V4Artifacts] = None
        self._model: Optional[ContextAwareGATv2_GRU_V4] = None

    def available(self) -> bool:
        req = [
            "meta.joblib",
            "graph.pt",
            "stop_id_to_idx.joblib",
            "scalers.joblib",
            "model.pt",
        ]
        return all(os.path.exists(os.path.join(self.artifacts_dir, r)) for r in req)

    def load(self) -> None:
        if not self.available():
            raise FileNotFoundError(f"Missing V4 artifacts in: {self.artifacts_dir}")

        meta = joblib.load(os.path.join(self.artifacts_dir, "meta.joblib"))
        graph = torch.load(os.path.join(self.artifacts_dir, "graph.pt"), map_location="cpu")
        stop_id_to_idx = joblib.load(os.path.join(self.artifacts_dir, "stop_id_to_idx.joblib"))
        scalers = joblib.load(os.path.join(self.artifacts_dir, "scalers.joblib"))

        seq_len = int(meta["seq_len"])
        context_seq_cols = list(meta["context_seq_cols"])
        gru_hidden = int(meta["gru_hidden"])
        dropout = float(meta["dropout"])

        graph_x = graph["x"].float()
        graph_edge_index = graph["edge_index"].long()

        self._artifacts = V4Artifacts(
            seq_len=seq_len,
            context_seq_cols=context_seq_cols,
            stop_id_to_idx={str(k): int(v) for k, v in stop_id_to_idx.items()},
            graph_x=graph_x.to(self.device),
            graph_edge_index=graph_edge_index.to(self.device),
            scaler_lag=scalers["scaler_lag"],
            scaler_roll=scalers["scaler_roll"],
            scaler_delta=scalers["scaler_delta"],
            scaler_prog=scalers["scaler_prog"],
        )

        self._model = ContextAwareGATv2_GRU_V4(
            num_node_features=int(self._artifacts.graph_x.shape[1]),
            num_seq_features=len(self._artifacts.context_seq_cols),
            gru_hidden=gru_hidden,
            dropout=dropout,
        ).to(self.device)

        state = torch.load(os.path.join(self.artifacts_dir, "model.pt"), map_location="cpu")
        self._model.load_state_dict(state)
        self._model.eval()

    def predict_stop_delay_seconds(
        self,
        *,
        stop_id: str,
        hour_sin: float,
        hour_cos: float,
        day_sin: float,
        day_cos: float,
        prev_delay_seconds: float,
        rolling_prev_delay_seconds: float,
        time_delta_seconds: float,
        progress: float,
    ) -> Optional[float]:
        if self._model is None or self._artifacts is None:
            self.load()

        assert self._model is not None
        assert self._artifacts is not None

        stop_id = str(stop_id)
        idx = self._artifacts.stop_id_to_idx.get(stop_id)
        if idx is None:
            return None

        # Scale features exactly as in training.
        prev_scaled = _scale_1d(self._artifacts.scaler_lag, _safe_float(prev_delay_seconds))
        roll_scaled = _scale_1d(self._artifacts.scaler_roll, _safe_float(rolling_prev_delay_seconds))
        delta_scaled = _scale_1d(self._artifacts.scaler_delta, _safe_float(time_delta_seconds))
        prog_scaled = _scale_1d(self._artifacts.scaler_prog, _safe_float(progress))

        # Build a single-timestep window and left-pad to seq_len.
        feats = np.array(
            [[
                _safe_float(hour_sin),
                _safe_float(hour_cos),
                _safe_float(day_sin),
                _safe_float(day_cos),
                prev_scaled,
                roll_scaled,
                delta_scaled,
                prog_scaled,
            ]],
            dtype=np.float32,
        )

        seq_len = self._artifacts.seq_len
        seq = np.zeros((1, seq_len, feats.shape[1]), dtype=np.float32)
        seq[0, -1:, :] = feats

        seq_t = torch.from_numpy(seq).to(self.device)
        lengths_t = torch.tensor([1], dtype=torch.long, device=self.device)
        stop_idx_t = torch.tensor([idx], dtype=torch.long, device=self.device)

        with torch.no_grad():
            out = self._model(self._artifacts.graph_x, self._artifacts.graph_edge_index, seq_t, lengths_t, stop_idx_t)
            return float(out.squeeze().item())
