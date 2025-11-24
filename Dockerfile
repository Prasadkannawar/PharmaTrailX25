FROM python:3.10-slim

# Prevent Python from writing .pyc files and force stdout flushing
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps for psycopg2 and xgboost build
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE 10000

# Start FastAPI (Phase 3 integrated API); honor PORT if provided (Render, etc.)
CMD ["sh", "-c", "uvicorn phase3_integrated_api:app --host 0.0.0.0 --port ${PORT:-10000}"]
