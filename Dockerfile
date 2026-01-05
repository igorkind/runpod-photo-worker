# Base Image
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY builder.py .
COPY handler.py .

# Run builder to download models
RUN python builder.py && \
    rm -rf /root/.cache/huggingface/hub && \
    rm -rf /root/.cache/pip

# Command to run the handler
CMD [ "python", "-u", "handler.py" ]
