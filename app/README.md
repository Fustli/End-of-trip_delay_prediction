# Stop-to-stop routing (fastest scheduled) + current delay

This folder contains a minimal web UI to:
- pick 2 GTFS stops on a map
- route via the **fastest scheduled journey** (direct or with transfers)
- show a **current delay estimate** if a GTFS-RT VehiclePositions feed is configured
- show **V4 GNN model predicted delays** for each leg

## Quick Start

### Option 1: Automated Setup (Recommended)

From repo root:
```bash
./scripts/setup.sh
```

This will:
1. Create a Python virtual environment
2. Install all dependencies
3. Set up PostgreSQL via Docker (if Docker is available)
4. Load GTFS data into the database

### Option 2: Manual Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the server (uses CSV files)
uvicorn app.backend.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` in your browser.

## Database Setup (Optional but Recommended)

Using PostgreSQL significantly speeds up data loading and routing queries.

### Using Docker:
```bash
docker run -d --name gtfs-postgres \
    -e POSTGRES_USER=gtfs_user \
    -e POSTGRES_PASSWORD=gtfs_password \
    -e POSTGRES_DB=gtfs_db \
    -p 5432:5432 \
    postgres:15

# Load GTFS data
python scripts/load_gtfs_to_db.py
```

### Environment Variables:
```bash
# Database connection
DATABASE_URL=postgresql://gtfs_user:gtfs_password@localhost:5432/gtfs_db

# Enable database usage (default: false, uses CSV files)
USE_DATABASE=true
```

## Realtime configuration (optional)

Set:
- `GTFS_RT_VEHICLE_POSITIONS_URL` to your VehiclePositions protobuf URL

Optionally:
- `GTFS_RT_API_KEY` and `GTFS_RT_API_KEY_HEADER` if your provider requires a header

Timezone for schedule interpretation:
- `GTFS_TIMEZONE` (default: `Europe/Budapest`)
