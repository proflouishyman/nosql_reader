# Stage 1: Build
FROM python:3.10-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 
ENV PYTHONUNBUFFERED=1          

WORKDIR /app

# 1. Install uv for extremely fast builds
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. Install dependencies BEFORE copying code (Layer Caching)
COPY requirements.txt .  
RUN uv pip install --system --no-cache -r requirements.txt

# 3. Pre-download spaCy (Only re-runs if requirements change)
RUN python -m spacy download en_core_web_lg

# 4. Copy code LAST
COPY . .

# Stage 2: Runtime
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1  
ENV PYTHONUNBUFFERED=1          
WORKDIR /app

# Copy only the site-packages and binaries from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

RUN chmod +x /app/entrypoint.sh /app/bootstrap_data.sh

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "main.py"]