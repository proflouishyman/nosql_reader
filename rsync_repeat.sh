#!/bin/bash

# Set variables
PASSWORD="Ba[2more"  # Replace with your actual password
SOURCE="lhyman6@192.168.86.30:~/coding/nosql_reader/archives/"
DESTINATION="./archives/"
RETRY_DELAY=5  # Time to wait before retrying in seconds

# Function to perform rsync with password
perform_rsync() {
    sshpass -p "$PASSWORD" rsync -avz --no-perms "$SOURCE" "$DESTINATION"
}

# Infinite loop to retry rsync
while true; do
    perform_rsync

    # Check the exit status of rsync
    if [ $? -eq 0 ]; then
        echo "Rsync completed successfully."
        break  # Exit the loop if rsync was successful
    else
        echo "Rsync failed. Retrying in $RETRY_DELAY seconds..."
        sleep $RETRY_DELAY  # Wait before retrying
    fi
done
