#!/bin/bash
# Start multiple Ollama instances on different ports

echo "Starting Ollama instances..."

# Kill any existing Ollama processes
pkill ollama

# Start instances
CUDA_VISIBLE_DEVICES=0 OLLAMA_HOST=127.0.0.1:11434 ollama serve &
sleep 5
CUDA_VISIBLE_DEVICES=0 OLLAMA_HOST=127.0.0.1:11435 ollama serve &
sleep 5
CUDA_VISIBLE_DEVICES=0 OLLAMA_HOST=127.0.0.1:11436 ollama serve &
sleep 5
CUDA_VISIBLE_DEVICES=0 OLLAMA_HOST=127.0.0.1:11437 ollama serve &

echo "Waiting for instances to start..."
sleep 10

# Load the model on each instance
for port in 11434 11435 11436 11437; do
    echo "Loading model on port $port..."
    curl -X POST http://localhost:$port/api/generate -d '{
        "model": "llama3.2-vision:11b",
        "prompt": "test",
        "stream": false
    }'
done

echo "All instances started and models loaded!"
