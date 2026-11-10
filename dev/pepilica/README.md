# Pepilica's TabPack experiments

CUDA kernels and benchmark helpers live in `src/tabpack_experimental/`.
This is a development package used alongside the root TabPackage checkout and its CUDA/benchmark extras, not an independent implementation of TabPack.

From the repository root, install the personal package with:

```bash
uv sync --locked --all-extras --group pepilica
```

Imports use `tabpack_experimental`, for example:

```python
from tabpack_experimental.compact_mlp import CompactFixedSlotMLP
from tabpack_experimental.active_fixed_slot import ActiveFixedSlotMLP
```

Configs and runners are under `configs/` and `scripts/` in this personal directory.
Run entry points from the repository root with module syntax, for example:

```bash
uv run --locked --all-extras --group pepilica python -m dev.pepilica.scripts.profile_maps_epoch3 --help
```

Tests are retained locally in `dev/pepilica/tests/` and ignored by Git. Run them with:

```bash
uv run --locked --all-extras --group pepilica python -m pytest -c dev/pepilica/pytest.ini dev/pepilica/tests
```

Benchmarks can run without the local test directory; test-source snapshots are optional.

Benchmark reports and implementation notes are in [docs/](docs/). Start with the
[backend guide](docs/ragged-backend.md) or the
[Maps-routing memory analysis](docs/benchmark-maps-phase-memory.md).
