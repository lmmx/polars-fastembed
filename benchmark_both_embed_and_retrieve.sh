hyperfine \
  './original/.venv/bin/python original/demo.py' \
  './rewrite/.venv/bin/python rewrite/demo.py' \
  -n python-plugin \
  -n rust-rewrite \
  --warmup 10
