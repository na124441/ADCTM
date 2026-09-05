# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install curl for health checking and cleanup apt cache
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Enforce OpenEnv compliance validation during the image build process
RUN openenv validate

# Expose port for Hugging Face Spaces
EXPOSE 7860

# Define environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=7860
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Container healthcheck testing the root endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# MANDATORY: Run the application on port 7860 using uvicorn
CMD ["python", "app.py"]

