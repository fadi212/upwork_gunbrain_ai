# Dockerfile

# Use Python 3.11.5-slim as the base image
FROM python:3.11.5-slim

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies (if any)
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Create and set the working directory
WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip

# Copy the requirements.txt file first for better caching
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app

# Expose port 8080 for FastAPI
EXPOSE 8080

# Create a non-root user for better security with a home directory
RUN addgroup --system appgroup && \
    adduser --system --ingroup appgroup --home /home/appuser appuser && \
    mkdir -p /app/unstructured_json_dir && \
    chown -R appuser:appgroup /app/unstructured_json_dir && \
    mkdir -p /home/appuser/.cache/unstructured/ingest/pipeline && \
    chown -R appuser:appgroup /home/appuser/.cache

# Set HOME environment variable
ENV HOME=/home/appuser

# Switch to the non-root user
USER appuser

# Define the default command to run the FastAPI app using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
