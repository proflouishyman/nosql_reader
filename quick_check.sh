#!/bin/bash

# Quick Diagnostic Script for Image Ingestion Issues
# Run this script to identify and fix common problems

set -e

echo "=================================================="
echo "  🔍 Image Ingestion Quick Diagnostic"
echo "=================================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Test 1: Check if Ollama is running
echo "Test 1: Checking Ollama service..."
if curl -s --max-time 3 http://localhost:11434/api/tags > /dev/null 2>&1; then
    print_success "Ollama is running on localhost:11434"
    OLLAMA_RUNNING=true
    OLLAMA_URL="http://localhost:11434"
elif curl -s --max-time 3 http://host.docker.internal:11434/api/tags > /dev/null 2>&1; then
    print_success "Ollama is accessible via host.docker.internal:11434"
    OLLAMA_RUNNING=true
    OLLAMA_URL="http://host.docker.internal:11434"
else
    print_error "Ollama is not accessible"
    OLLAMA_RUNNING=false
    echo ""
    echo "To fix this:"
    echo "  1. Start Ollama: ollama serve"
    echo "  2. Or install Ollama from: https://ollama.ai"
    echo ""
    exit 1
fi
echo ""

# Test 2: Check available models
echo "Test 2: Checking for vision models..."
MODELS=$(curl -s $OLLAMA_URL/api/tags | grep -o '"name":"[^"]*"' | cut -d'"' -f4)

if [ -z "$MODELS" ]; then
    print_error "No models found in Ollama"
    echo "Install a vision model:"
    echo "  ollama pull llama3.2-vision:latest"
    exit 1
fi

print_success "Available models:"
echo "$MODELS" | while read -r model; do
    echo "  - $model"
done

# Check for vision models
if echo "$MODELS" | grep -q "vision\|llava\|bakllava"; then
    print_success "Vision-capable models found"
else
    print_warning "No vision models found"
    echo ""
    echo "Install a vision model:"
    echo "  ollama pull llama3.2-vision:latest"
    echo "  OR"
    echo "  ollama pull llava:latest"
    echo ""
    exit 1
fi
echo ""

# Test 3: Check Docker container
echo "Test 3: Checking Docker container..."
if docker compose ps | grep -q "app.*running"; then
    print_success "Flask app container is running"
else
    print_error "Flask app container is not running"
    echo ""
    echo "Start the container:"
    echo "  docker compose up -d"
    exit 1
fi
echo ""

# Test 4: Check mounted directories
echo "Test 4: Checking mounted directories..."
if docker compose exec app test -d /data/archives 2>/dev/null; then
    print_success "Archives directory is mounted"
    
    IMAGE_COUNT=$(docker compose exec app find /data/archives -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) 2>/dev/null | wc -l)
    
    if [ "$IMAGE_COUNT" -gt 0 ]; then
        print_success "Found $IMAGE_COUNT image files"
    else
        print_warning "No image files found in /data/archives"
        echo ""
        echo "Add images to: ../archives/"
        exit 1
    fi
else
    print_error "Archives directory not accessible"
    echo ""
    echo "Check docker-compose.yml mount configuration"
    exit 1
fi
echo ""

# Test 5: Test actual connection from container
echo "Test 5: Testing Ollama connection from container..."
if docker compose exec app curl -s --max-time 3 http://host.docker.internal:11434/api/tags > /dev/null 2>&1; then
    print_success "Container can reach Ollama via host.docker.internal:11434"
elif docker compose exec app curl -s --max-time 3 http://172.17.0.1:11434/api/tags > /dev/null 2>&1; then
    print_success "Container can reach Ollama via 172.17.0.1:11434"
    print_warning "Update your config to use http://172.17.0.1:11434"
else
    print_error "Container CANNOT reach Ollama"
    echo ""
    echo "Try these fixes:"
    echo "  1. Use host network: Add 'network_mode: host' to app service in docker-compose.yml"
    echo "  2. Use bridge IP: Update OLLAMA_BASE_URL to http://172.17.0.1:11434"
    echo "  3. Restart containers: docker compose restart"
    exit 1
fi
echo ""

# Summary
echo "=================================================="
echo "  ✅ ALL DIAGNOSTICS PASSED!"
echo "=================================================="
echo ""
echo "Your system appears to be configured correctly."
echo ""
echo "Next steps:"
echo "  1. Open: http://localhost:5000/settings"
echo "  2. Go to 'Data Ingestion' section"
echo "  3. Verify Ollama Base URL: $OLLAMA_URL"
echo "  4. Click 'Refresh models' to populate dropdown"
echo "  5. Select a vision model (llama3.2-vision or llava)"
echo "  6. Click 'Scan for new images'"
echo ""
echo "If you still see 0 JSON files generated, run:"
echo "  docker compose logs app | grep -i error"
echo ""