hyperfine \
  './original/.venv/bin/python original/embed_demo.py' \
  './rewrite/.venv/bin/python rewrite/embed_demo.py' \
  -n python-plugin \
  -n rust-rewrite \
  --warmup 10