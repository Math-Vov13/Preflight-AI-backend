# Stage 1: Builder - Préparation des dépendances et modèles
FROM python:3.13-slim-bookworm AS builder

# Installer uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copier les fichiers de configuration des dépendances
COPY pyproject.toml /app/pyproject.toml
COPY uv.lock /app/uv.lock

# Build deps for native wheels
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN uv sync --locked

# Stage 2: Final - Image optimisée pour l'exécution
FROM python:3.13-slim-bookworm AS final

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copier l'environnement virtuel du builder
# COPY --from=builder /app/.venv /app/.venv

# Copier les fichiers de configuration et le code source
COPY pyproject.toml /app/pyproject.toml
COPY uv.lock /app/uv.lock
# COPY .env /app/.env
COPY ./src /app/src

ENV PATH="/app/.venv/bin:${PATH}"

EXPOSE 8080

CMD ["uv", "run", "src/server.py", "--host", "0.0.0.0", "--port", "8080"]
