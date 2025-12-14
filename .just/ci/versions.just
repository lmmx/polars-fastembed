[working-directory: 'polars-fastembed']
_min_version:
    #!/usr/bin/env -S bash -euo pipefail
    uvx --from=toml-cli toml get --toml-path=pyproject.toml project.requires-python | sed 's/>=//;s/"//g'

_avail_pythons:
    uv python list --all-versions --output-format json

_avail_cpythons:
    #!/usr/bin/env -S bash -euo pipefail
    MIN_VERSION=$(just _min_version)
    # numpy doesn't build on 3.15 yet
    MAX_VERSION="3.14"
    just _avail_pythons | jq -c --arg min "$MIN_VERSION" --arg max "$MAX_VERSION" '
      ($min | split(".") | {major: .[0]|tonumber, minor: .[1]|tonumber}) as $minv |
      ($max | split(".") | {major: .[0]|tonumber, minor: .[1]|tonumber}) as $maxv |
      [.[] | {v:.version_parts, t:.variant, impl:.implementation, full:.version}]
      | unique_by([.v.minor,.t])
      | map(select(
         (.impl == "cpython") and
         (.v.major == 3) and
         (.v.minor >= $minv.minor) and
         (.v.minor <= $maxv.minor)
        ))
      | map(select(.t != "freethreaded"))
      | map("\(.v.major).\(.v.minor)")'

# PyPy disabled - builds never finish
_avail_pypys_DISABLED:
    #!/usr/bin/env -S bash -euo pipefail
    MIN_VERSION=$(just _min_version)
    # pyo3 0.23.4 supports PyPy up to 3.10
    MAX_PYPY="3.10"
    just _avail_pythons | jq -c --arg min "$MIN_VERSION" --arg max "$MAX_PYPY" '
      ($min | split(".") | {major: .[0]|tonumber, minor: .[1]|tonumber}) as $minv |
      ($max | split(".") | {major: .[0]|tonumber, minor: .[1]|tonumber}) as $maxv |
      [.[] | {v:.version_parts, t:.variant, impl:.implementation}]
      | unique_by([.v.minor,.impl])
      | map(select(
          (.impl == "pypy") and
          (.v.major == 3) and
          (.v.minor >= $minv.minor) and
          (.v.minor <= $maxv.minor)
        ))
      | map("pypy\(.v.major).\(.v.minor)")'

_avail_pypys:
    #!/usr/bin/env -S bash -euo pipefail
    echo "[]"

ci_version_matrix:
    #!/usr/bin/env -S bash -euo pipefail
    CPYTHON_VERSIONS=$(just _avail_cpythons)
    PYPY_VERSIONS=$(just _avail_pypys)

    # versions=$(jq -nc --argjson a "$CPYTHON_VERSIONS" '$a')
    versions=$(jq -nc --argjson a "$CPYTHON_VERSIONS" --argjson b "$PYPY_VERSIONS" '$a + $b')
    echo "python-versions=$versions"

ci_interpreter_list:
    #!/usr/bin/env -S bash -euo pipefail
    CPYTHON=$(just _avail_cpythons | jq -r 'join(" ")')
    PYPY=$(just _avail_pypys | jq -r 'join(" ")')
    echo "$CPYTHON $PYPY"
