# Use an official PyTorch base image with CUDA support
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set the working directory
WORKDIR /app

# Copy the local files to the container
COPY . /app

# Install necessary system packages, including tkinter
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-opencv python3-tk && \
    rm -rf /var/lib/apt/lists/*

# Install required Python libraries
RUN pip install --no-cache-dir \
    torchvision==0.15.2 \
    numpy==1.23.5 \
    Pillow==9.4.0 \
    argparse

# Expose the container's working directory as a volume (optional, for data/model sharing)
VOLUME /app

# Default command to run when the container starts
CMD ["bash"]
