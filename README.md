# Deep Learning Class (VITMMA19) Project Work template

## Project Details

### Project Information

- **Selected Topic**: End-of-trip delay prediction
- **Student Name**: Norbert Bendegúz Hasznos
- **Aiming for +1 Mark**: No

### Solution Description

We predict end-of-trip delay in seconds from GTFS-RT vehicle position snapshots.

The workflow is:
- Data cleansing: filter invalid/outlier delay values and GPS points outside the Budapest bounding box.
- Baselines: linear regression and a sequence of Random Forest models with increasingly informative features.
- GNN models: graph-based regressors that use a stop-to-stop graph inferred from trip sequences.

Model families included:
- Baseline models (scikit-learn): DummyRegressor, LinearRegression, and RandomForestRegressor variants.
- GNN Model V1: spatial graph convolution baseline.
- GNN Model V2: context-aware Graph Attention Network (GAT) with real-time lag features and cyclical time embeddings.
- GNN Model V3: temporal + spatial model (sequence windowing + GNN) implemented in the V3 notebook.

### Extra Credit Justification

[If you selected "Yes" for Aiming for +1 Mark, describe here which specific part of your work (e.g., innovative model architecture, extensive experimentation, exceptional performance) you believe deserves an extra mark.]

## Logging Requirements

This repository follows the grading requirement of logging only essential information, with no `print()` statements and no emojis.

### Where logs go

- Notebooks and scripts use `src/utils.py::setup_logger` to log to both stdout and a file under `log/`.
- Expected notebook log files:
    - `log/data_cleansing.log`
    - `log/data_visualization.log`
    - `log/baseline-log.txt`
    - `log/gnn_model_v1_training.log`
    - `log/gnn_model_v2_training.log`
    - `log/gnn_model_v3_training.log`

### What must be logged

- Data: input paths used, successful load confirmation, row counts (before/after filtering), and split sizes.
- Configuration/hyperparameters actually used (e.g., batch size, learning rate, number of epochs, scheduler settings).
- Training: per-epoch training and validation metrics (at minimum MAE; optionally RMSE/R²).
- Evaluation: final test-set metrics (MAE/RMSE/R²).
- Artifacts: any plots saved under `plots/` with their output paths.

### Docker Instructions

This project is containerized using Docker. Follow the instructions below to build and run the solution.
[Adjust the commands that show how do build your container and run it with log output.]

#### Build

Run the following command in the root directory of the repository to build the Docker image:

```bash
docker build -t dl-project .
```

#### Run

To run the solution, use the following command. You must mount your local data directory to `/app/data` inside the container.

**To capture the logs for submission (required), redirect the output to a file:**

```bash
docker run -v /absolute/path/to/your/local/data:/app/data dl-project > log/run.log 2>&1
```

*   Replace `/absolute/path/to/your/local/data` with the actual path to your dataset on your host machine that meets the [Data preparation requirements](#data-preparation).
*   The `> log/run.log 2>&1` part ensures that all output (standard output and errors) is saved to `log/run.log`.
*   The container is configured to run every step (data preprocessing, training, evaluation, inference).


### File Structure and Functions

The repository is structured as follows:

- **`src/`**: Contains the source code for the machine learning pipeline.
    - `01-data-scraper.py`: Fetches/updates GTFS-RT vehicle positions (data acquisition).
    - `02-data-cleanser.py`: Cleans and filters raw vehicle positions into a modeling-ready CSV.
    - `03-training.py`: Training script for GNN Model V4 (GATv2Conv + GRU).
    - `04-evaluation.py`: Evaluation script that computes metrics on the test set.
    - `05-inference.py`: Inference script for running the model on new/unseen data.
    - `config.py`: Central configuration for paths and hyperparameters.
    - `utils.py`: Logger setup utilities (file + stdout).
    - `run.sh`: Shell script that orchestrates the full ML pipeline.

- **`notebook/`**: Contains Jupyter notebooks for analysis and experimentation.
    - `01_data-cleansing.ipynb`: Data quality filtering and export to cleaned CSV.
    - `02_data-visualization.ipynb`: Exploratory data analysis and plots.
    - `03_baseline-model.ipynb`: Baseline models (Linear Regression, Random Forest) and diagnostics.
    - `04_gnn-modelV1.ipynb`: Spatial GNN baseline (GCN).
    - `05_gnn-modelV2.ipynb`: Context-aware GAT with lag features.
    - `06_gnn-modelV3.ipynb`: Temporal + Spatial model (GATConv + GRU).
    - `07_gnn-modelV4.ipynb`: Final model (GATv2Conv + GRU) - best performance.

- **`models/`**: Trained model artifacts.
    - `gnn_v4/`: GNN V4 model weights, graph, scalers, and metadata.

- **`log/`**: Contains log files.
    - `run.log`: Recommended submission log path for script-based runs.
    - `gnn_model_v4_training.log`: Training logs.
    - `gnn_model_v4_evaluation.log`: Evaluation logs.

- **`plots/`**: Saved diagnostic plots.
    - `gnn_model_v4_diagnostics.png`: Evaluation plots (pred vs actual, residuals, error distribution).

- **`data/`**: Data files (mounted via Docker volume).
    - `vehicle_positions.csv`: Raw scraped data.
    - `vehicle_positions_cleaned.csv`: Cleaned data ready for modeling.
    - `predictions.csv`: Model predictions output.
    - `gtfs/`: Static GTFS schedule files.

- **Root Directory**:
    - `Dockerfile`: Configuration file for building the Docker image with the necessary environment and dependencies.
    - `requirements.txt`: List of Python dependencies required for the project.
    - `README.md`: Project documentation and instructions.