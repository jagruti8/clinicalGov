FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-eval.txt ./
# Set INSTALL_EVAL=1 to include Inspect (evaluation tasks): docker build --build-arg INSTALL_EVAL=1 .
ARG INSTALL_EVAL=0
RUN if [ "$INSTALL_EVAL" = "1" ]; then pip install --no-cache-dir -r requirements-eval.txt; \
    else pip install --no-cache-dir -r requirements.txt; fi

COPY . .

VOLUME ["/app/chroma_db", "/app/ctgov_cache"]
EXPOSE 8000

# CLI by default:  docker run IMAGE "question"      |  docker run IMAGE --smoke
# REST API:        docker run -p 8000:8000 IMAGE --serve
ENTRYPOINT ["python", "main.py"]
