FROM python:3.10

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p static/uploads

EXPOSE 7860

CMD ["gunicorn", "--workers=1", "--timeout=120", "--bind=0.0.0.0:7860", "app:app"]