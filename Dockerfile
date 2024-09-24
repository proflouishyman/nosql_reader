# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Ensure environment variables are set for Flask
ENV FLASK_APP=app.py
ENV FLASK_ENV=development  

# Expose port 5000 for Flask to bind to
EXPOSE 5000

# Define the command to run the app
CMD ["flask", "run", "--host=0.0.0.0"]
