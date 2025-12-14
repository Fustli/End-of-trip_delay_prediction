#!/bin/bash
#
# Setup script for End-of-trip Delay Prediction project
#
# This script:
# 1. Creates a Python virtual environment
# 2. Installs dependencies
# 3. Sets up PostgreSQL (via Docker if available)
# 4. Loads GTFS data into the database
#
# Usage:
#   ./scripts/setup.sh [--skip-db] [--skip-venv]
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "========================================"
echo "End-of-trip Delay Prediction Setup"
echo "========================================"
echo ""

# Parse arguments
SKIP_DB=false
SKIP_VENV=false
for arg in "$@"; do
    case $arg in
        --skip-db)
            SKIP_DB=true
            ;;
        --skip-venv)
            SKIP_VENV=true
            ;;
    esac
done

# Step 1: Create virtual environment
if [ "$SKIP_VENV" = false ]; then
    echo -e "${GREEN}Step 1: Setting up Python virtual environment...${NC}"
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        echo "  Created virtual environment"
    else
        echo "  Virtual environment already exists"
    fi

    # Activate and install dependencies
    source venv/bin/activate
    echo "  Installing dependencies..."
    pip install --upgrade pip > /dev/null
    pip install -r requirements.txt
    echo -e "  ${GREEN}Dependencies installed!${NC}"
else
    echo -e "${YELLOW}Skipping virtual environment setup${NC}"
    source venv/bin/activate 2>/dev/null || true
fi

# Step 2: Setup PostgreSQL
if [ "$SKIP_DB" = false ]; then
    echo ""
    echo -e "${GREEN}Step 2: Setting up PostgreSQL database...${NC}"
    
    # Check if Docker is available
    if command -v docker &> /dev/null; then
        # Check if container already exists
        if docker ps -a --format '{{.Names}}' | grep -q '^gtfs-postgres$'; then
            # Check if it's running
            if docker ps --format '{{.Names}}' | grep -q '^gtfs-postgres$'; then
                echo "  PostgreSQL container is already running"
            else
                echo "  Starting existing PostgreSQL container..."
                docker start gtfs-postgres
            fi
        else
            echo "  Creating PostgreSQL container with Docker..."
            docker run -d --name gtfs-postgres \
                -e POSTGRES_USER=gtfs_user \
                -e POSTGRES_PASSWORD=gtfs_password \
                -e POSTGRES_DB=gtfs_db \
                -p 5432:5432 \
                postgres:15
            echo "  Waiting for PostgreSQL to start..."
            sleep 5
        fi
        
        # Create .env file if it doesn't exist
        if [ ! -f ".env" ]; then
            echo 'DATABASE_URL=postgresql://gtfs_user:gtfs_password@localhost:5432/gtfs_db' > .env
            echo 'USE_DATABASE=true' >> .env
            echo "  Created .env file with database configuration"
        fi
        
        DB_AVAILABLE=true
    else
        echo -e "  ${YELLOW}Docker not found. Skipping database setup.${NC}"
        echo "  You can manually set up PostgreSQL and run:"
        echo "    python scripts/load_gtfs_to_db.py"
        DB_AVAILABLE=false
    fi
    
    # Load GTFS data if database is available
    if [ "$DB_AVAILABLE" = true ]; then
        echo ""
        echo -e "${GREEN}Step 3: Loading GTFS data into database...${NC}"
        
        # Check if data already loaded
        source venv/bin/activate
        export DATABASE_URL="postgresql://gtfs_user:gtfs_password@localhost:5432/gtfs_db"
        
        python3 -c "
from app.backend.database import check_database_populated
if check_database_populated():
    print('ALREADY_LOADED')
else:
    print('NEED_LOAD')
" 2>/dev/null || echo "NEED_LOAD" > /tmp/db_status.txt

        if python3 -c "from app.backend.database import check_database_populated; exit(0 if check_database_populated() else 1)" 2>/dev/null; then
            echo "  Database already contains GTFS data"
        else
            echo "  Loading GTFS data (this may take a few minutes)..."
            python3 scripts/load_gtfs_to_db.py --drop-existing
        fi
    fi
else
    echo -e "${YELLOW}Skipping database setup${NC}"
fi

echo ""
echo "========================================"
echo -e "${GREEN}Setup complete!${NC}"
echo "========================================"
echo ""
echo "To start the application:"
echo "  1. Activate the virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Start the backend server:"
echo "     python -m uvicorn app.backend.main:app --host 127.0.0.1 --port 8000"
echo ""
echo "  3. Open in browser:"
echo "     http://127.0.0.1:8000/"
echo ""
if [ "$DB_AVAILABLE" = true ] 2>/dev/null; then
    echo "Database is running. Set USE_DATABASE=true to use it."
fi
