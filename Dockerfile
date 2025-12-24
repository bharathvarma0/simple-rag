FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directory for data
RUN mkdir -p data/pdfs vector_store

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose API port
EXPOSE 8000

# Persist data


# Default command to run the API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
