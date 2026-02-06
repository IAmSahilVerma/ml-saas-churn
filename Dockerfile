# Use NVIDIA CUDA base image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# Install Python + pip
RUN apt-get update && apt-get install -y python3.10 python3-pip

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Install PyTorch + torchvision + torchaudio (CUDA 11.8)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

COPY . .

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]