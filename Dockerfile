# syntax=docker/dockerfile:1

# ============================================
# Stage 1: Builder
# ============================================
FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libopenblas-dev \
    liblapack-dev \
    pkg-config \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Build FAISS from source (static)
RUN git clone --depth 1 https://github.com/facebookresearch/faiss.git /faiss
WORKDIR /faiss

RUN cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DFAISS_ENABLE_PYTHON=OFF \
    -DBUILD_TESTING=OFF \
    -DFAISS_ENABLE_GPU=OFF \
    -DFAISS_ENABLE_RAFT=OFF \
    -DFAISS_ENABLE_BLAS=ON \
    -DCMAKE_INSTALL_PREFIX=/usr/local && \
    cmake --build build --parallel $(nproc) && \
    cmake --install build

# Copy only the REST API project
COPY . /app
WORKDIR /app

# Build the REST API directly (no dependency on build.sh)
RUN rm -rf build && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    cmake --build . --parallel $(nproc)

# ============================================
# Stage 2: Runtime (minimal)
# ============================================
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy only the final binary
COPY --from=builder /app/build/faiss_rest_api /usr/local/bin/faiss_rest_api

# Create non-root user
RUN useradd -m -u 1001 faissuser
USER faissuser

EXPOSE 8080

ENTRYPOINT ["faiss_rest_api"]
CMD ["8080"]