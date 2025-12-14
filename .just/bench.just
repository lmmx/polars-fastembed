[working-directory: 'benchmark']
bench *args:
    uv run --frozen benchmark {{args}}
