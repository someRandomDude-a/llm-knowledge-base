# docker/Dockerfile
FROM python:3.14-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxml2-dev libxslt-dev gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY code/ code/

EXPOSE 8005

WORKDIR /app/code
ENV WORKERS 2
CMD ["sh", "-c", "python -m healthcheck & uvicorn api:app --host 0.0.0.0 --port 8005 --workers ${WORKERS}"]