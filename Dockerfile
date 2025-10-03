# Base image with PyTorch + CUDA
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

# 2. Set working directory inside the container
WORKDIR /app

# 3. Install system-level dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy Python dependencies and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy application code and run scripts
COPY ./src .

# 6. Make the run script executable
RUN chmod +x run.sh

# 7. Default command when container starts
CMD ["bash", "run.sh"]

