# AegisFabric network/training lab: privileged Linux container hosting network namespaces,
# FRRouting instances, tc/netem links, Gloo training ranks and the Rust checkpoint engine.
# Build: make lab-image   (package-index URL is passed as a BuildKit secret, never baked in)
FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
      iproute2 iputils-ping iperf3 nftables procps psmisc tcpdump ethtool \
      frr frr-pythontools xfsprogs e2fsprogs util-linux bash ca-certificates curl \
      build-essential pkg-config clang libclang-dev cmake git \
    && rm -rf /var/lib/apt/lists/*

# Rust toolchain for ckpt/ (built inside the lab so io_uring code paths are exercised on Linux).
ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo PATH=/opt/cargo/bin:$PATH
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv
WORKDIR /opt/aegis
COPY docker/requirements-lab.txt /opt/aegis/requirements-lab.txt
RUN --mount=type=secret,id=uvindex,env=UV_INDEX_URL \
    uv venv /opt/venv -p /usr/local/bin/python3.12 && \
    VIRTUAL_ENV=/opt/venv uv pip install --no-deps --index-url https://download.pytorch.org/whl/cpu torch && \
    VIRTUAL_ENV=/opt/venv uv pip install -r /opt/aegis/requirements-lab.txt \
      filelock sympy networkx jinja2 fsspec typing-extensions setuptools
ENV PATH=/opt/venv/bin:$PATH PYTHONPATH=/work AEGIS_HOST_LABEL=lab-container
WORKDIR /work
CMD ["bash"]
