## Release process

To release a new version, download the wheels generated from the rewrite to `dist/`
then run PDM publish with `--no-build`. The README will come from the artifacts you download,
not the local one.

- `pre-commit run all-files`
- `git push`
- `gh run download -p wheel*`
- `mv wheel*/* dist/ && rm -rf wheel* && pdm publish --no-build`
