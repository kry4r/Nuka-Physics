# Contributing to Nuka Physics

Thank you for your interest in contributing to **Nuka Physics**, a GPU-resident,
strongly deterministic, differentiable physics engine for robotics simulation
and reinforcement learning.

This document describes how to build, test, and submit changes. By contributing
you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## Project license

Nuka Physics is **dual-licensed**: the [GNU AGPL v3.0](LICENSE) (open source) or a
commercial license (see [LICENSING.md](LICENSING.md)). See [NOTICE](NOTICE) for
third-party attributions. Do not introduce code under licenses incompatible with
the AGPL. By submitting a contribution you license it to the project under the
AGPL (inbound = outbound); there is no separate CLA at this time.

## Architecture

Before proposing larger changes, read the architecture docs:

- Architecture docs: [`docs/architecture/`](docs/architecture/)

## Building

The engine targets **CUDA 12.8** with **g++-10** (the C++11 ABI is set to `1`;
this is the ABI the Python bindings expect). The CUDA build directory is
**`build-cuda128`** — always use this directory, never the legacy `build/`
(which historically used a different/incorrect CUDA toolkit).

### C++ engine (GPU)

```bash
export CC=gcc-10 CXX=g++-10
cmake -S . -B build-cuda128 \
  -DNK_BUILD_TESTS=ON \
  -DNK_REQUIRE_CUDA=ON \
  -DNK_PHYSICS_BACKEND=CUDA
cmake --build build-cuda128 -j
```

A CUDA-capable GPU is required to build and run the production physics path.

### Python bindings

The Python extension (`_nuka_ext`) is built against the engine in
`build-cuda128`:

```bash
pip install -e python
```

See [`python/README.md`](python/README.md) for the environment variables
(`NUKA_BUILD_DIR`, `CUDA_LIB_DIR`, `CMAKE_CXX_COMPILER`) that point the build at
the engine library and the matching ABI compiler.

## Determinism rule (hard requirement)

Nuka enforces **D1 strong determinism**: a simulation must be **bit-exact across
two runs**, with no float atomics and a fixed reduction order in physics paths.

- **Do not use floating-point `atomicAdd` (or other float atomics) in physics
  code paths.** This breaks the D1 determinism contract.
- This is enforced by the physics-smell lint, which **must pass** on every PR:

  ```bash
  python tools/lint/physics_smell.py
  ```

  The lint also blocks hot-path `cudaMalloc`, `throw` across the C ABI, STL in
  public headers, and missing generated-file headers. See
  [`tools/lint/README.md`](tools/lint/README.md) for the full pattern set and
  the scoped allowlist policy. The allowlist is only for documented migration
  debt or validation code — never use it to land new physics code.

## Generated code: DO NOT EDIT

Files under **`src/codegen/generated/**`** are produced by the codegen step and
must not be hand-edited. They are required to start with the header
`// GENERATED — DO NOT EDIT` (the lint enforces this). To change generated
output, change the codegen inputs/templates and regenerate via the build, then
commit the regenerated files.

## Test gates

A pull request must pass the following:

### C++ tests (ctest)

```bash
ctest --test-dir build-cuda128 --output-on-failure
```

### Python tests (pytest)

```bash
pytest python/tests/
```

### Known pre-existing failures (NOT regressions)

Two oracle tests are **documented, pre-existing, carry-forward failures** and
are **not** regressions. They reflect an ABA-vs-MuJoCo/MJX oracle gap documented
since sim-validation issue #43, orthogonal to current work:

- `V01FoundationE2E.Phase6CudaAbaMatchesGo2AndH1MuJoCoOracle` (#38)
- `FeatherstoneOracle.RandomSampleGoldensMatchCudaAba` (#206)

Your PR may leave only these two tests failing. It must not introduce any new
test failures, and it must not "fix" them by weakening the oracle. Do not modify
protected golden artifacts without an explicit, reviewed protected-change
proposal.

## Submitting a pull request

1. Branch from the active development branch (not `master` directly unless told).
2. Keep changes focused; reference the relevant issue or master-plan item.
3. Run, locally, before opening the PR:
   - `python tools/lint/physics_smell.py` (must be clean)
   - `ctest --test-dir build-cuda128 --output-on-failure` (only the 2 known
     fails above may fail)
   - `pytest python/tests/`
4. Confirm you have **not** edited `src/codegen/generated/**` or protected
   goldens, and that you introduced **no new float atomics** in physics paths.
5. Open the PR and fill in the pull request template.

## Reporting bugs and requesting features

Use the GitHub issue templates (bug report / feature request). For security
vulnerabilities, **do not** open a public issue — follow [SECURITY.md](SECURITY.md).
