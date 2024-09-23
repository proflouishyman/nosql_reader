#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Function to check if a command exists
command_exists () {
    command -v "$1" >/dev/null 2>&1
}

# 1. Check and Install Docker
if ! command_exists docker; then
    echo "Docker not found. Installing Docker..."

    # Update package index
    sudo apt-get update

    # Install prerequisites
    sudo apt-get install -y ca-certificates curl gnupg lsb-release

    # Add Docker’s official GPG key
    sudo mkdir -p /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

    # Set up the Docker repository
    echo \
      "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
      $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

    # Install Docker Engine
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

    echo "Docker installed successfully."
else
    echo "Docker is already installed."
fi

# Start and enable Docker service
sudo systemctl start docker
sudo systemctl enable docker

# 2. Check and Install Docker Compose (if needed)
if ! docker compose version >/dev/null 2>&1; then
    if command_exists docker-compose; then
        echo "Docker Compose (v1) is installed."
    else
        echo "Docker Compose not found. Installing Docker Compose..."

        # Get the latest Docker Compose version number
        COMPOSE_VERSION=$(curl -s https://api.github.com/repos/docker/compose/releases/latest | grep tag_name | cut -d '"' -f 4)

        # Download Docker Compose binary
        sudo curl -L "https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

        # Apply executable permissions
        sudo chmod +x /usr/local/bin/docker-compose

        # Verify installation
        docker-compose --version

        echo "Docker Compose installed successfully."
    fi
else
    echo "Docker Compose is already installed."
fi

# 3. Create Docker Compose Configuration for MongoDB
DOCKER_COMPOSE_FILE="docker-compose.yml"

if [ ! -f "$DOCKER_COMPOSE_FILE" ]; then
    echo "Creating $DOCKER_COMPOSE_FILE..."

    cat <<EOF > $DOCKER_COMPOSE_FILE
services:
  mongodb:
    image: mongo:4.4
    container_name: mongodb
    ports:
      - "27017:27017"
    volumes:
      - ./data:/data/db
    environment:
      MONGO_INITDB_ROOT_USERNAME: admin
      MONGO_INITDB_ROOT_PASSWORD: secret
EOF

    echo "$DOCKER_COMPOSE_FILE created."
else
    echo "$DOCKER_COMPOSE_FILE already exists."
fi

# 4. Start MongoDB Container
echo "Starting MongoDB container..."

sudo docker compose up -d

echo "MongoDB is up and running."

# 5. Verify MongoDB Container Status
echo "Verifying MongoDB container status..."

sudo docker compose ps

echo "Setup complete."
