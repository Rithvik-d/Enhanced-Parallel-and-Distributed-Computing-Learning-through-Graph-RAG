# Dockerfile for CDER GraphRAG System
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables (will be overridden by .env)
ENV PYTHONUNBUFFERED=1

# Expose port if needed for future web interface
EXPOSE 8000

# Default command
CMD ["python", "main.py"]

