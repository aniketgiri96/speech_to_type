# Use a robust base image for building
FROM ubuntu:22.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    python3 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Clone the specific llama.cpp fork and branch for LFM audio support
# PR #18641: https://github.com/ggml-org/llama.cpp/pull/18641
RUN git clone https://github.com/ggml-org/llama.cpp.git && \
    cd llama.cpp && \
    git fetch origin pull/18641/head:pr-18641 && \
    git checkout pr-18641

# Configure CMake for CPU build (default)
# Note: For GPU support, we would need a different base image (e.g., nvidia/cuda) and flags
WORKDIR /app/llama.cpp
RUN cmake -B build -DGGML_CUDA=OFF -DLLAMA_CURL=OFF

# Build the server binary
RUN cmake --build build --config Release --target llama-liquid-audio-server -j $(nproc)

# Expose the server port
EXPOSE 8142

# Create a directory for models (to be mounted)
RUN mkdir -p /app/models

# Set the entrypoint to the server binary
ENTRYPOINT ["./build/bin/llama-liquid-audio-server"]

# Default command arguments (can be overridden in docker-compose or CLI)
# Assumes models are mounted to /app/models
CMD ["-m", "/app/models/LFM2.5-Audio-1.5B-Q4_0.gguf", \
     "-mm", "/app/models/mmproj-LFM2.5-Audio-1.5B-Q4_0.gguf", \
     "-mv", "/app/models/vocoder-LFM2.5-Audio-1.5B-Q4_0.gguf", \
     "--tts-speaker-file", "/app/models/tokenizer-LFM2.5-Audio-1.5B-Q4_0.gguf", \
     "--host", "0.0.0.0", \
     "--port", "8142"]
