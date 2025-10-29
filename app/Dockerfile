# ===========================
# Dockerfile: RTSP Analyzer Worker (Optimized)
# docker build -t shinwoohipo/worker-alarm-panel:1.0.0 .
# ===========================

# 1️⃣ Build stage
FROM python:3.10-slim AS build
WORKDIR /app

# 최소 빌드 deps 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 설치 (headless OpenCV)
COPY requirements_slim.txt .
RUN pip install --upgrade pip
RUN pip install --user --no-cache-dir -r requirements_slim.txt

# 소스 코드 복사
COPY . .

# ===========================
# 2️⃣ Final stage (runtime only)
FROM python:3.10-slim
WORKDIR /app

# 환경 변수
ENV TZ=Asia/Seoul
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=-1
ENV PATH=/root/.local/bin:$PATH

# 빌드 스테이지에서 설치된 패키지만 복사
COPY --from=build /root/.local /root/.local
COPY --from=build /app /app

