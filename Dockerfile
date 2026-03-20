# HallucinationGuard-Env Dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Install the package
RUN pip install -e .

# Expose the port
EXPOSE 8000

# Run the server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
