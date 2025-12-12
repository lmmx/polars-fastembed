[working-directory: 'rewrite']
_min_version:
    #!/usr/bin/env -S bash -euo pipefail
    uvx --from=toml-cli toml get --toml-path=pyproject.toml project.requires-python | sed 's/>=//;s/"//g'

_avail_pythons:
    uv python list --all-versions --output-format json

_avail_cpythons:
    #!/usr/bin/env -S bash -euo pipefail
    MIN_VERSION=$(just _min_version)
    # Excluding 3.15.0a2 specifically as numpy won't build on it (needed for pyarrow)
    just _avail_pythons | jq -c --arg min "$MIN_VERSION" '
      ($min | split(".") | {major: .[0]|tonumber, minor: .[1]|tonumber}) as $minv |
      [.[] | {v:.version_parts, t:.variant, impl:.implementation, full:.version}]
      | unique_by([.v.minor,.t])
      | map(select(
         (.impl == "cpython") and
         (.v.major == 3) and
         (.v.minor >= $minv.minor) and
         (.full != "3.15.0a2")
        ))
      | map(select(.t != "freethreaded")) # Delete this line to include FT variants
      | map("\(.v.major).\(.v.minor)\(if .t=="freethreaded" then "t" else "" end)")'

_avail_pypys:
    #!/usr/bin/env -S bash -euo pipefail
    MIN_VERSION=$(just _min_version)
    just _avail_pythons | jq -c --arg min "$MIN_VERSION" '
      ($min | split(".") | {major: .[0]|tonumber, minor: .[1]|tonumber}) as $minv |
      [.[] | {v:.version_parts, t:.variant, impl:.implementation}]
      | unique_by([.v.minor,.impl])
      | map(select(
          (.impl == "pypy") and
          (.v.major == 3) and
          (.v.minor >= $minv.minor)
        ))
      | map("pypy-\(.v.major).\(.v.minor)")'

ci_version_matrix:
    #!/usr/bin/env -S bash -euo pipefail
    CPYTHON_VERSIONS=$(just _avail_cpythons)
    # PYPY_VERSIONS=$(just _avail_pypys)

    # versions=$(jq -nc --argjson a "$CPYTHON_VERSIONS" --argjson b "$PYPY_VERSIONS" '$a + $b')
    versions=$(jq -nc --argjson a "$CPYTHON_VERSIONS" '$a')
    echo "python-versions=$versions"
