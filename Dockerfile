FROM nvcr.io/nvidia/pytorch:24.01-py3

# Set environment variables for non-interactive installations and unbuffered Python output.
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv (the Python package manager) globally.
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /app

COPY pyproject.toml ./


COPY uv.lock ./

RUN uv install --system

# Copy the rest of your application code into the container.
# This assumes your `train.py` and other scripts/modules are in the root of your project directory.
COPY . .

CMD ["uv", "run", "train.py"]