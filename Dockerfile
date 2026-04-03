# Use slim Python image
FROM python:3.11-slim

# Set working directory to project root
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc python3-dev libpq-dev libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project into container
COPY . .

# Set PYTHONPATH so Python can find src module
ENV PYTHONPATH=/app

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI using module syntax
CMD ["python", "-m", "src.api.main"]