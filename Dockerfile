FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    unzip \
    && curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip \
    && unzip /tmp/awscliv2.zip -d /tmp \
    && /tmp/aws/install \
    && rm -rf /tmp/awscliv2.zip /tmp/aws \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml .
RUN uv sync

COPY flows/ ./flows/
COPY lib/ ./lib/
COPY aoi/ ./aoi/
COPY config.gers.yaml .
COPY run_gersite.py .

ENV PATH="/app/.venv/bin:${PATH}"
ENV LIFELINE_STORAGE_ROOT=/data

CMD ["python", "run_gersite.py", "--aoi", "florida"]
