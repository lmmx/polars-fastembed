hyperfine \
  './original/.venv/bin/python original/demo.py' \
  './polars-fastembed/.venv/bin/python polars-fastembed/demo.py' \
  -n python-plugin \
  -n rust-polars-fastembed \
  --warmup 10
