# Stage 1: Build
FROM python:3.10-slim AS builder

# Set environment variables to prevent .pyc files and ensure output is flushed
ENV PYTHONDONTWRITEBYTECODE=1  # Prevents Python from writing .pyc files
ENV PYTHONUNBUFFERED=1          # Ensures logs are printed in real-time

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \  # Required for compiling some Python packages
    && rm -rf /var/lib/apt/lists/*  # Clean up to reduce image size

# Copy the requirements file and install Python dependencies
COPY requirements.txt .  # Copy only the requirements file to leverage caching
RUN pip install --upgrade pip && pip install --user -r requirements.txt

# Download spaCy models only if they are not already downloaded
RUN if [ ! -d "/root/.local/share/spacy/models/en_core_web_lg" ]; then \
        python -m spacy download en_core_web_lg; \
    fi && \
    if [ ! -d "/root/.local/share/spacy/models/en_core_web_trf" ]; then \
        python -m spacy download en_core_web_trf; \
    fi
# This avoids redundant downloads and saves time and bandwidth

# Copy project files into the container
COPY . .

# Copy spaCy models to ensure they are available in the runtime stage
RUN cp -r /root/.local/share/spacy/models /root/.local/share/spacy/models

# Stage 2: Runtime
FROM python:3.10-slim

# Set environment variables for the runtime stage
ENV PYTHONDONTWRITEBYTECODE=1  # Prevents Python from writing .pyc files
ENV PYTHONUNBUFFERED=1          # Ensures logs are printed in real-time

# Set the working directory inside the runtime container
WORKDIR /app

# Install only necessary system dependencies for the runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*  # Clean up to reduce image size

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
# Ensures the entrypoint script is executable and ready to run

# Set the entrypoint to the entrypoint script for container initialization
ENTRYPOINT ["/app/entrypoint.sh"]

# Command to run the Flask app
CMD ["python", "app.py"]
