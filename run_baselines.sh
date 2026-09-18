micromamba run -n r-baselines bash -c '
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

  exec uv run --frozen python r_baselines_fit.py "$@"
' run_baselines "$@"
