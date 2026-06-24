# Circuit DLLM — GPU contributor node image.
#
# Bundles the inference engine + its CUDA/PyTorch/AWQ deps so a GPU operator runs ONE
# container that joins the live mesh: detect GPU -> register with the coordinator
# (ed25519-signed) -> receive a layer slice -> pull only that slice's AWQ shards -> serve.
#
# The operator never installs Python/torch/CUDA — only the NVIDIA Container Toolkit on the
# host. Build + publish from a machine with engine access; contributors just `docker run`.
#
#   docker build -f docker/Dockerfile -t ghcr.io/circuit-llm/gpu-node:latest .
#
# Base: official PyTorch CUDA runtime (torch 2.4 + CUDA 12.1 + cuDNN9), matches the prod pods.
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DEBIAN_FRONTEND=noninteractive

# nvidia-smi (capacity detection) ships via the NVIDIA Container Toolkit at runtime; we only
# need a couple of host tools for the entrypoint.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Engine deps. gptqmodel provides the Marlin AWQ kernels (2.1x over bnb); cryptography backs
# the ed25519 node identity + the ChaCha20 data wire. Pinned-ish for reproducibility.
RUN pip install --upgrade pip && pip install \
        "transformers>=4.44,<4.49" \
        "huggingface_hub>=0.24" \
        "accelerate>=0.33" \
        "gptqmodel>=1.0" \
        "cryptography>=42" \
        "numpy<2"

# Bake the engine (build context = the circuit-dllm repo root).
WORKDIR /opt/circuit-engine
COPY engine/  ./engine/
COPY deploy/  ./deploy/
COPY scripts/ ./scripts/
COPY docker/entrypoint.sh /usr/local/bin/circuit-gpu-node
RUN chmod +x /usr/local/bin/circuit-gpu-node \
    && python -c "import ast,glob,sys; [ast.parse(open(f).read()) for f in glob.glob('engine/*.py')]; print('engine parses')"

# Persistent state lives here — mount a named volume so the node KEEPS its identity and its
# downloaded shards across container recreation (re-registering as a NEW node each restart
# would churn the mesh and re-download GBs).
ENV CIRCUIT_ENGINE_DIR=/opt/circuit-engine \
    CIRCUIT_STATE_DIR=/var/lib/circuit \
    HF_HOME=/var/lib/circuit/hf-cache \
    CIRCUIT_NODE_KEY_FILE=/var/lib/circuit/node_key.hex \
    CIRCUIT_MODEL=Qwen/Qwen2.5-72B-Instruct-AWQ \
    CIRCUIT_MODEL_FP=qwen2.5-72b-awq \
    CIRCUIT_AWQ_SHARDS=circuitllmdev/qwen2.5-72b-awq-shards \
    CIRCUIT_PORT=19210
VOLUME /var/lib/circuit
EXPOSE 19210

ENTRYPOINT ["/usr/local/bin/circuit-gpu-node"]
