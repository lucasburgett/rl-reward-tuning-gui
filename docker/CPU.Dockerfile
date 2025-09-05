# docker/CPU.Dockerfile
FROM python:3.11-slim

# Faster, quieter, predictable builds
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=0 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    PYTHONHASHSEED=0

# System deps for SB3/Gym & optional video/export
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    swig \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Leverage layer caching: install deps first
COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy project files
COPY . .
ENV PYTHONPATH=/app

# Default to an interactive shell; examples are in README
CMD ["/bin/bash"]