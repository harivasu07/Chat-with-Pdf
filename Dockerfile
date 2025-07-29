# Use the official Python 3.12 slim image as the base
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt first to leverage Docker caching
COPY requirements.txt /app/

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app

# Expose the port that Chainlit will use
EXPOSE 8000

# Run the Chainlit app
CMD ["chainlit", "run", "your_script_name.py", "-w"]
