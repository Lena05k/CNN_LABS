FROM python:3.10-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/raw data/processed \
    lab1/outputs/checkpoints \
    lab1/outputs/figures \
    lab1/outputs/tables \
    lab1/outputs/metrics

CMD ["python", "lab1/scripts/run_lab1.py"]
