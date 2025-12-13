FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps (lightweight; keeps matplotlib/opencv-headless happy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Copy the whole repo so paths like data/, log/, plots/, notebook/ exist
COPY . .

RUN chmod +x src/run.sh

# Run the full ML pipeline (preprocessing, training, evaluation, inference)
CMD ["bash", "src/run.sh"]

