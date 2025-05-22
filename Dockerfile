FROM nvcr.io/nvidia/pytorch:24.01-py3

# Set environment variables for non-interactive installations and unbuffered Python output.
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY pyproject.toml ./


COPY uv.lock ./

RUN uv sync --locked

COPY . .

CMD ["uv", "run", "train.py"]