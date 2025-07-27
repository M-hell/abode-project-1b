# Use Python 3.11 slim image for smaller size
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for PyMuPDF and other libraries
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model download script
COPY download_model.py .

# Download and cache the AI model (this requires internet during build)
RUN python download_model.py

# Copy the main processing script
COPY process_pdfs.py .

# Create input and output directories
RUN mkdir -p /app/input /app/output

# Make the script executable (optional, but good practice)
RUN chmod +x process_pdfs.py

# Set the default command
CMD ["python", "process_pdfs.py"]