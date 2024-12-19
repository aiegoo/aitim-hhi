# Use an official PyTorch base image
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set the working directory
WORKDIR /app

# Copy the local files to the container
COPY . /app

# Install necessary system packages, including OpenCV, tkinter, PaddleOCR dependencies, and fonts
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-opencv python3-tk ttf-nanum && \
    rm -rf /var/lib/apt/lists/*

# Set NanumGothic.ttf as the default font in the system
RUN ln -s /usr/share/fonts/truetype/nanum/NanumGothic.ttf /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf

# Install required Python libraries, including PaddlePaddle (CPU version)
RUN pip install --no-cache-dir \
    paddlepaddle==2.5.2 \
    paddleocr==2.6.1.3 \
    torchvision==0.15.2 \
    numpy==1.23.5 \
    Pillow==9.4.0

# Expose the container's working directory as a volume (optional, for data/model sharing)
VOLUME /app

# Default command to run when the container starts
CMD ["bash"]
