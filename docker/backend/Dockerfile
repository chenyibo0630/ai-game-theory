# Backend: runs simulations and writes them to MySQL.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# ca-certificates is needed for TLS to LLM providers. No curl — kept lean.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY db ./db
COPY config.yaml.example ./config.yaml.example

# Default command waits for MySQL then runs one simulation using the mounted
# config.yaml; override via `command:` in docker-compose for custom workloads.
COPY docker/backend/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Run as non-root so a container escape doesn't yield host root.
RUN useradd --system --uid 1001 --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app
USER appuser

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "-m", "src.main", "--config", "/app/config.yaml", "--storage", "mysql"]
