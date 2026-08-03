FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    awscli \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml .
RUN uv sync --no-dev

# Pre-install DuckDB extensions at build time so Fargate containers don't
# need to reach out to DuckDB's CDN at runtime.
RUN /app/.venv/bin/python -c "\
import duckdb; \
con = duckdb.connect(':memory:'); \
con.sql(\"INSTALL spatial\"); \
con.sql(\"INSTALL httpfs\"); \
con.close()"

COPY . .

# Patch config.gers.yaml for container environment at build time
RUN sed -i \
    -e 's|/Users/praveenaparimi/data/gers|/data/gers|g' \
    -e 's|/Users/praveenaparimi/Documents/GitHub/GERSite/aoi/|/app/aoi/|g' \
    -e 's|memory_limit: "8GB"|memory_limit: "24GB"|g' \
    /app/config.gers.yaml

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV S3_BUCKET=geocube-files-prod
ENV AWS_DEFAULT_REGION=us-east-1

CMD ["python", "/app/run_gersite.py", "--help"]
