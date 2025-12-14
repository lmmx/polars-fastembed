## Release process

Preamble:

- `pre-commit run all-files`
- `maturin develop --release`
- `git push`

Release:

```sh
test -z "$(compgen -G 'wheel*/')" || {
  echo "Please delete the wheel*/ dirs:" >&2
  ls wheel*/ -1d >&2
  false
} && \
gh run watch \
  "$(gh run list -L 1 --json databaseId --jq .[0].databaseId)" \
  --exit-status && \
gh run download \
  "$(gh run list -L 1 --json databaseId --jq .[0].databaseId)" \
  -p wheel* && \
rm -rf dist/ && \
mkdir dist/ && \
mv wheel*/* dist/ && \
rm -rf wheel*/ && \
pdmpublish --no-build
```

  - (`pdmpublish` is an alias for `pdm publish ...`)
