# Training Dockerfile for Briscola RL PPO Agent
# Trains the model against LLM opponent (Ollama)

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ ./src/
COPY training/ ./training/
COPY briscola_rl/ ./briscola_rl/
COPY baml_client/ ./baml_client/
COPY baml_src/ ./baml_src/
COPY pyproject.toml .

# Create directories for checkpoints and models
RUN mkdir -p checkpoints models

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Default command - can be overridden
CMD ["python", "-m", "training.train_vs_llm", \
     "--num-updates", "1000", \
     "--episodes-per-update", "10", \
     "--eval-interval", "10", \
     "--save-dir", "/app/checkpoints"]
