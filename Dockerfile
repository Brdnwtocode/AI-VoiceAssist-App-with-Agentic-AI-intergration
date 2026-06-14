FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (e.g. for build tools or potential network checks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source directory and config file
# Note: credentials/ folder is NOT copied — use GOOGLE_CREDENTIALS_JSON env var on EC2
COPY src/ ./src/
COPY litellm_config.yaml .

# Expose FastAPI's default port mapped in main.py
EXPOSE 8000

# Start the application using python src/main.py
CMD ["python", "src/main.py"]
