# Stage 1: Build
FROM python:3.10-slim AS builder

# Set environment variables to prevent .pyc files and ensure output is flushed
ENV PYTHONDONTWRITEBYTECODE=1 
ENV PYTHONUNBUFFERED=1          

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install Python dependencies
COPY requirements.txt .  
RUN pip install --upgrade pip && pip install --user -r requirements.txt

# Create the directory for spaCy models to ensure it exists
RUN mkdir -p /root/.local/share/spacy/models

# Download spaCy models only if they are not already downloaded
RUN if [ ! -d "/root/.local/share/spacy/models/en_core_web_lg" ]; then \
        python -m spacy download en_core_web_lg; \
    fi && \
    if [ ! -d "/root/.local/share/spacy/models/en_core_web_trf" ]; then \
        python -m spacy download en_core_web_trf; \
    fi

# Copy project files into the container
COPY . .

# Stage 2: Runtime
FROM python:3.10-slim

# Set environment variables for the runtime stage
ENV PYTHONDONTWRITEBYTECODE=1  
ENV PYTHONUNBUFFERED=1          

# Set the working directory inside the runtime container
WORKDIR /app

# Install only necessary system dependencies for the runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*  

# Copy installed Python packages and spaCy models from the builder stage
COPY --from=builder /root/.local /root/.local
COPY --from=builder /root/.local/share/spacy/models /root/.local/share/spacy/models

# Set PATH to include user-installed binaries
ENV PATH=/root/.local/bin:$PATH

# Copy project files from the builder stage
COPY --from=builder /app /app

# Copy and set permissions for the entrypoint script
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Set the entrypoint to the entrypoint script for container initialization
ENTRYPOINT ["/app/entrypoint.sh"]

# Command to run the Flask app
CMD ["python", "app.py"]
