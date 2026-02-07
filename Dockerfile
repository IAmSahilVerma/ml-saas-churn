FROM python:3.10-slim

# Prevent python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies (needed for xgboost + torch)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Copy project
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
