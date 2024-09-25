#Dockerfile
# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY app/requirements.txt .

# Upgrade pip and install the Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY ./app .


COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh




# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=development
ENV FLASK_DEBUG=1

# Make port 5000 available outside the container
EXPOSE 5000

# Switch to the non-root user
# USER appuser


ENTRYPOINT ["/entrypoint.sh"]

# Define the command to run the application
CMD ["flask", "run", "--host=0.0.0.0"]