micromamba run -n r-baselines bash -c '
  export RPY2_CFFI_MODE=ABI
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

  exec uv run --with debugpy \
    python -m debugpy \
    --listen 127.0.0.1:5678 \
    --wait-for-client \
    r_baselines_fit.py
'