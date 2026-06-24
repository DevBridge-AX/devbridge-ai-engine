FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 의존성만 먼저 복사 (코드 변경 시에도 pip 캐시 재사용)
COPY pyproject.toml .

# BuildKit 캐시 마운트 (Docker 기본 활성화)
# 최초 1회만 다운로드, 이후 빌드는 캐시 사용 = 네트워크 불필요
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install .

COPY . .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
