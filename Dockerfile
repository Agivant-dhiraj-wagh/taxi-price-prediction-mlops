# Use a base image with Python
FROM python:3.9-slim-buster

# Set the working directory
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the trainer and pipeline code
COPY trainer/ trainer/
COPY pipeline/ pipeline/

# Set environment variables (optional, but good practice)
ENV PYTHONUNBUFFERED=1

# Command to run your training script (this will be overridden by Vertex AI later, but good for local testing)
CMD ["python", "trainer/task.py"]