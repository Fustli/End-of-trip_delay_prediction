# Deep Learning Class (VITMMA19) Project Work

## Project Details

### Project Information

- **Selected Topic**: End-of-trip delay prediction
- **Student Name**: Norbert Bendegúz Hasznos
- **Aiming for +1 Mark**: No

### Solution Description

We predict end-of-trip delay in seconds from GTFS-RT vehicle position snapshots collected from the Budapest public transit system (BKK).

**Problem**: Given real-time vehicle position data (lat/lon, timestamp, trip_id, stop_id), predict how late (or early) a vehicle will be at the end of its trip. This is a regression task where accurate predictions can help passengers plan journeys and operators optimize schedules.

**Data Pipeline**:
1. **Data acquisition**: GTFS-RT vehicle positions scraped via BKK API (`src/01-data-scraper.py`)
2. **Data cleansing**: Filter invalid/outlier delay values and GPS points outside Budapest (`src/02-data-cleanser.py`)
3. **Dataset size**: 23.9M raw rows → 15.1M cleaned rows (37% outliers removed)

**Model Evolution**:
- **Baseline models** (scikit-learn): DummyRegressor, LinearRegression, and RandomForestRegressor variants with increasingly informative features
- **GNN Model V1**: Spatial graph convolution baseline (GCNConv)
- **GNN Model V2**: Context-aware Graph Attention Network (GATConv) with real-time lag features
- **GNN Model V3**: Temporal + spatial model (GATConv + GRU)
- **GNN Model V4** (Final): GATv2Conv + GRU - best performance

**Architecture (GNN V4)**:
- **Spatial branch**: Two GATv2Conv layers learn stop embeddings from transit network topology (5,212 nodes, 38,207 edges)
- **Temporal branch**: GRU encodes windowed context (last 12 stops: delays, time, progress)
- **Fusion**: Concatenate spatial + temporal embeddings → MLP (256→128→64→1) → delay prediction
- **Parameters**: 174,209 trainable parameters

**Training Methodology**:
- **Optimizer**: Adam with learning rate 0.003 and ReduceLROnPlateau scheduler (patience=4, factor=0.5)
- **Batch size**: 4,096 samples
- **Epochs**: 50 (with early stopping based on validation MAE)
- **Loss function**: L1 Loss (Mean Absolute Error)
- **Data split**: 70% train / 10% validation / 20% test (chronological split)
- **Hardware**: NVIDIA RTX 4060 GPU (~47 seconds/epoch)

**Results**:

| Metric | GNN Model V4 | Baseline (Context-Aware RF) | Improvement |
|--------|--------------|----------------------------|-------------|
| MAE    | **37.21s**   | 43.18s                     | -13.8%      |
| RMSE   | **87.14s**   | 78.53s                     | +11.0%      |
| R²     | **0.8673**   | 0.8918                     | -2.7%       |

The GNN Model V4 achieves a **Mean Absolute Error of 37.21 seconds**, outperforming the best baseline model (Context-Aware Random Forest with 43.18s MAE) by approximately 6 seconds (13.8% relative improvement). While the baseline achieves a slightly higher R² (0.8918 vs 0.8673), the GNN model's lower MAE indicates better absolute prediction accuracy, which is more relevant for practical delay estimation.

### Data Preparation

The raw data consists of GTFS-RT vehicle positions scraped from the BKK (Budapest Public Transit) API.

**Required**: Place your raw vehicle positions CSV at `data/vehicle_positions.csv` with the following columns:
- `timestamp`: ISO datetime string
- `trip_id`: Unique trip identifier
- `delay_seconds`: Delay in seconds (target variable)
- `latitude`, `longitude`: GPS coordinates
- `last_stop_id` or `stop_id`: Stop identifier

The Docker pipeline will automatically clean and preprocess the data, creating `vehicle_positions_cleaned.csv`.

**Cleansing rules applied automatically**:
- Remove rows with missing delay values
- Remove exact zero delays (API artifacts)
- Remove unrealistic delays (< -1800s or > 1800s)
- Remove GPS points outside Budapest bounding box (47.30-47.65°N, 18.90-19.35°E)

### Inference on New Data

To run predictions on new/unseen data after training:

1. Place your inference data at `data/inference.csv` with the same columns as the training data:
   - `timestamp`, `trip_id`, `latitude`, `longitude`, `last_stop_id` (or `stop_id`)
   - `delay_seconds` is optional for inference (will be ignored if present)

2. Run the Docker container - it will automatically detect `inference.csv`, **cleanse it using the same pipeline**, and generate predictions:
   ```bash
   docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/log:/app/log -v $(pwd)/plots:/app/plots dl-project
   ```

3. The cleansed inference data is saved to `data/inference_cleaned.csv` and predictions to `data/predictions.csv` with a `predicted_delay` column.

**Note**: If no `inference.csv` is provided, the pipeline runs inference on the cleaned training data by default.

### Docker Instructions

This project is containerized using Docker with **CUDA GPU support**. The base image includes PyTorch with CUDA 12.4 for accelerated training.

#### Prerequisites

- Docker installed
- For GPU support: NVIDIA GPU with drivers + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

#### Build

Run the following command in the root directory of the repository to build the Docker image:

```bash
docker build -t dl-project .
```

#### Run (GPU - Recommended)

For fast training with GPU acceleration:

```bash
docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/log:/app/log -v $(pwd)/plots:/app/plots dl-project 2>&1 | tee log/run.log
```

#### Run (CPU Only)

If no GPU is available, the model will automatically use CPU (slower):

```bash
docker run -v $(pwd)/data:/app/data -v $(pwd)/log:/app/log -v $(pwd)/plots:/app/plots dl-project 2>&1 | tee log/run.log
```

**Notes**:
- The `2>&1 | tee log/run.log` ensures logs are both displayed and saved to file.
- The container runs the full pipeline: data preprocessing → training → evaluation → inference.
- GPU training is ~10-20x faster than CPU for this model.

#### Quick Test (Local)

```bash
# Build
docker build -t dl-project .

# Run with local data directory mounted
docker run -v $(pwd)/data:/app/data -v $(pwd)/log:/app/log -v $(pwd)/plots:/app/plots dl-project
```


### File Structure and Functions

The repository is structured as follows:

- **`src/`**: Contains the source code for the machine learning pipeline.
    - `01-data-scraper.py`: Fetches/updates GTFS-RT vehicle positions (data acquisition).
    - `02-data-cleanser.py`: Cleans and filters raw vehicle positions into a modeling-ready CSV.
    - `03-training.py`: Training script for GNN Model V4 (GATv2Conv + GRU).
    - `04-evaluation.py`: Evaluation script that computes metrics on the test set.
    - `05-inference.py`: Inference script for running the model on new/unseen data.
    - `config.py`: Central configuration for paths and hyperparameters (includes `GNN_V4_NUM_EPOCHS`, etc.).
    - `models.py`: Shared model class definitions.
    - `utils.py`: Logger setup utilities (file + stdout).
    - `run.sh`: Shell script that orchestrates the full DL pipeline.

- **`notebook/`**: Contains Jupyter notebooks for analysis and experimentation.
    - `01_data-cleansing.ipynb`: Data quality filtering and export to cleaned CSV.
    - `02_data-visualization.ipynb`: Exploratory data analysis and plots.
    - `03_baseline-model.ipynb`: Baseline models (Linear Regression, Random Forest) and diagnostics.
    - `04_gnn-modelV1.ipynb`: Spatial GNN baseline (GCNConv).
    - `05_gnn-modelV2.ipynb`: Context-aware GAT with lag features (GATConv).
    - `06_gnn-modelV3.ipynb`: Temporal + Spatial model (GATConv + GRU).
    - `07_gnn-modelV4.ipynb`: Final model (GATv2Conv + GRU) - best performance.

- **`models/`**: Trained model artifacts.
    - `gnn_v4/`: GNN V4 model weights, graph, scalers, and metadata.

- **`log/`**: Contains log files.
    - `run.log`: Combined pipeline log (required for submission).

- **`plots/`**: Saved diagnostic plots.
    - `gnn_model_v4_diagnostics.png`: Evaluation plots (pred vs actual, residuals, error distribution).

- **`data/`**: Data files (mounted via Docker volume).
    - `vehicle_positions.csv`: Raw scraped data.
    - `vehicle_positions_cleaned.csv`: Cleaned data ready for modeling.
    - `predictions.csv`: Model predictions output.
    - `gtfs/`: Static GTFS schedule files.

- **`app/`**: FastAPI web application (optional, for ML-as-a-service).
    - `backend/`: API endpoints and inference logic.
    - `frontend/`: Simple HTML frontend.
    - *Status: Like the buses it predicts, this feature is perpetually "arriving in 5 minutes." The backend exists, the frontend exists, but connecting them remains a task for a future developer who has more coffee and fewer deadlines. Consider it a monument to ambition.*

- **Root Directory**:
    - `Dockerfile`: Configuration file for building the Docker image.
    - `requirements.txt`: List of Python dependencies required for the project.
    - `README.md`: Project documentation and instructions.