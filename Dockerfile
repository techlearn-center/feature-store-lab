FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create directories for data
RUN mkdir -p /app/data/raw /app/data/processed /app/data/registry

# Default command: run the feature server
CMD ["uvicorn", "src.serving.feature_server:app", "--host", "0.0.0.0", "--port", "8000"]
