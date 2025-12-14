FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:/app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (torch already in base image)
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir torch-geometric \
    && pip install --no-cache-dir pandas>=2.2.0 scikit-learn>=1.4.0 joblib>=1.4.0 \
    matplotlib>=3.8.0 seaborn>=0.13.0 \
    fastapi>=0.115.0 "uvicorn[standard]>=0.32.0" pydantic>=2.0.0 \
    sqlalchemy>=2.0.0 psycopg2-binary>=2.9.0 \
    gtfs-realtime-bindings>=1.0.0 protobuf>=4.0.0 pytz>=2024.1 \
    requests>=2.31.0 python-dotenv>=1.0.0 tqdm>=4.66.0

# Copy source code
COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/

# Create necessary directories
RUN mkdir -p /app/data /app/log /app/plots

RUN chmod +x src/run.sh

# Run the full ML pipeline
CMD ["bash", "src/run.sh"]

