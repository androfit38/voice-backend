FROM python:3.11-slim

# Create a non-root user
RUN useradd --create-home --shell /bin/bash app

# Set working directory
WORKDIR /app

# Install system dependencies as root
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies as root (this is fine)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and change ownership to app user
COPY . .
RUN chown -R app:app /app

# Switch to non-root user
USER app

# Try to download model files (optional, will fallback to runtime download)
RUN python main.py download-files || echo "Model files will be downloaded at runtime"

# Expose port
EXPOSE 10000

# Run the application
CMD ["python", "main.py"]
