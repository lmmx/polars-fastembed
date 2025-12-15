import ".just/ci/architectures.just"
import ".just/ci/ort-providers.just"
import ".just/ci/versions.just"
import ".just/bench.just"
import ".just/commit.just"
import ".just/ship.just"

pc-fix:
  prek run --all-files
