# Lab image + Triton (for TRITON_INTERPRET correctness runs and offline CUDA-target IR compilation).
FROM aegis-lab:latest
RUN --mount=type=secret,id=uvindex,env=UV_INDEX_URL VIRTUAL_ENV=/opt/venv uv pip install "triton==3.8.0" "jax[cpu]==0.11.2"
