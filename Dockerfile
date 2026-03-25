FROM python:3.12-slim AS builder

WORKDIR /build
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir --prefix=/install .

# Install Playwright browser in builder to keep layers clean
RUN pip install playwright && playwright install chromium

# Runtime
FROM python:3.12-slim

# Minimal Chromium runtime deps for Crawl4AI
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 \
    libxkbcommon0 libxcomposite1 libxdamage1 libxrandr2 libgbm1 \
    libpango-1.0-0 libcairo2 libasound2 libxshmfence1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --create-home appuser

WORKDIR /app

COPY --from=builder /install /usr/local
COPY --from=builder /root/.cache/ms-playwright /home/appuser/.cache/ms-playwright
COPY src/ src/
COPY data/ data/

RUN chown -R appuser:appuser /app /home/appuser/.cache
USER appuser

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
