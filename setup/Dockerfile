# setup/Dockerfile
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Download spaCy models
RUN python -m spacy download en_core_web_lg && \
    python -m spacy download en_core_web_trf

# Copy project files
COPY . .

# Run setup scripts
# Run setup scripts

# Run setup scripts when the container starts
CMD ["sh", "-c", "python database_setup.py && python data_processing.py && python generate_unique_terms.py && python entity_linking.py"]
