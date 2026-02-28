FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .

# Upgrade pip
RUN pip install --upgrade pip

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Create uploads folder
RUN mkdir -p static/uploads

# Expose Hugging Face required port
EXPOSE 7860

# Start Flask using Gunicorn
CMD ["gunicorn", "--workers=2", "--timeout=120", "--bind=0.0.0.0:7860", "app:app"]
