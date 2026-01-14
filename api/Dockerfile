# API Dockerfile for Briscola RL FastAPI Backend
# Usage: docker build -t briscola-api .
#        docker run -p 8000:8000 briscola-api

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements from root directory
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir fastapi uvicorn[standard] pydantic

# Copy project files
COPY src/ ./src/
COPY api/ ./api/
COPY briscola_rl/ ./briscola_rl/
COPY baml_client/ ./baml_client/
COPY baml_src/ ./baml_src/

# Create necessary directories for models and checkpoints
RUN mkdir -p models checkpoints

# Expose port
EXPOSE 8000

# Health check using curl
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Run with uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]