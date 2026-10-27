# Build candidate: requires a linux/amd64 builder; no model weights in layers.
FROM vllm/vllm-openai@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
USER root
LABEL org.opencontainers.image.title="DeepSeek V4.1 1017 RTX96 inference candidate" \
      io.dsv41.runtime="vllm-0.32.0-sm120" \
      io.dsv41.validation="build-candidate; GPU validation required"
ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1
RUN test "$(dpkg --print-architecture)" = amd64 \
    && python3 -c 'import sys; assert sys.version_info[:2] == (3, 12)' \
    && apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates openssh-server iproute2 iperf3 python3-venv \
    && rm -f /etc/ssh/ssh_host_*_key /etc/ssh/ssh_host_*_key.pub \
    && rm -rf /var/lib/apt/lists/*
COPY build-kernels.py flashinfer-wheel-lock.json /opt/dsv41/
RUN python3 /opt/dsv41/build-kernels.py
# Operational tooling cannot change the inference interpreter's dependencies.
COPY requirements-vast.lock /opt/dsv41/requirements-vast.lock
RUN python3 -m venv /opt/vast-cli \
    && /opt/vast-cli/bin/python -m pip install --require-hashes --only-binary=:all: \
         -r /opt/dsv41/requirements-vast.lock \
    && /opt/vast-cli/bin/vastai destroy instance --help > /opt/dsv41/vast-destroy-help.txt \
    && grep -q -- '--yes' /opt/dsv41/vast-destroy-help.txt \
    && /opt/vast-cli/bin/python -m pip freeze --all > /opt/dsv41/vast-packages.txt
COPY runtime/ /opt/dsv41/runtime/
COPY init.sh verify-runtime.py source-manifest.json /opt/dsv41/
RUN python3 /opt/dsv41/verify-runtime.py --build-only \
    && python3 -m compileall -q /opt/dsv41/runtime \
    && bash -n /opt/dsv41/init.sh /opt/dsv41/runtime/launch_dsv41.sh \
    && python3 -m pip freeze --all > /opt/dsv41/runtime-packages.txt \
    && dpkg-query -W > /opt/dsv41/os-packages.txt
ENV HF_HOME=/workspace/hf-metadata \
    HF_XET_CHUNK_CACHE_SIZE_BYTES=0 TOKENIZERS_PARALLELISM=false \
    VAST_CLI=/opt/vast-cli/bin/vastai
WORKDIR /workspace
# Vast SSH mode replaces this entrypoint. Set onstart to bash /opt/dsv41/init.sh.
# Manual docker use is deliberately inert unless a command is supplied.
ENTRYPOINT ["/bin/bash"]
CMD ["-lc", "echo 'Image candidate: run /opt/dsv41/init.sh and complete preflight before weights.'"]
