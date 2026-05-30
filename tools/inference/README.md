# Go2 Policy C++ Inference Validation (`go2_policy_check`)

A small, **self-contained** libtorch harness that validates the **C++ inference
path** for the trained Go2 locomotion policy. It loads the TorchScript model,
runs it on **CPU**, and checks the 12-dim action outputs against a golden JSON
produced by the reference Python pipeline.

This is the standalone C++ inference validation that the future
engine-integrated inference (**Nuka RL inference**) will build on. Locking it
now proves the deployed `.pt` runs correctly under C++/libtorch and gives a
golden-backed regression check.

## What it checks (correctness bar)

For the model `motion.pt` (TorchScript MLP `48 -> 512 -> 256 -> 128 -> 12`, ELU):

1. **SHA-256 match** — the model file's SHA-256 must equal `golden["sha256_model"]`,
   so we know we loaded the exact asset the golden was generated from.
2. **Output match** — for each of the 4 golden cases, `max |y_cpp - y_golden|`
   must be `< 1e-4`. (CPU C++ reproduces the golden **bit-exactly**, max_err
   `0.0`.)

Exit code is `0` only if SHA matches **and** every case passes; otherwise `1`
(asset/IO errors return `2`).

## Files

- `go2_policy_check.cpp` — the harness (libtorch load + forward, a tiny
  self-contained SHA-256, and a minimal hand-rolled reader for the known-shape
  golden JSON — no third-party JSON dependency).
- `CMakeLists.txt` — its **own** `project()`. Intentionally **not** part of the
  main Nuka `CMakeLists.txt`.

## ABI note (critical) — pre-CXX11, do NOT link the engine

The on-box libtorch (pip, `torch 2.6.0+cu124`) is built with the **pre-CXX11
ABI**: `find_package(Torch)` reports

```
TORCH_CXX_FLAGS = -D_GLIBCXX_USE_CXX11_ABI=0
```

The CMakeLists applies `${TORCH_CXX_FLAGS}` to the target so this harness is
compiled `_GLIBCXX_USE_CXX11_ABI=0` to match libtorch.

**Do NOT link any Nuka engine library here.** The engine is built CXX11-ABI=1;
mixing the two ABIs clashes at the `std::string` / linker boundary. This harness
links **only** `${TORCH_LIBRARIES}` (libtorch + its CUDA deps, which are
resolved via libtorch's own rpath even though we run on CPU). When the real
engine-integrated inference is built, the engine side must be reconciled with
libtorch's ABI (e.g. an ABI-matched libtorch, or isolating the libtorch call
behind a C ABI shim) — this harness deliberately sidesteps that by being
standalone.

## Configure / build / run

Build in a **separate** build dir so it never touches `build-cuda128` or the
engine build. Point `Torch_DIR` at the conda torch CMake package.

```bash
TORCH_CMAKE=/root/miniconda3/envs/nuka-v03/lib/python3.10/site-packages/torch/share/cmake/Torch

cmake -S /root/Nuka-Physics/tools/inference \
      -B /root/Nuka-Physics/tools/inference/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_COMPILER=g++-10 \
      -DTorch_DIR=$TORCH_CMAKE

cmake --build /root/Nuka-Physics/tools/inference/build -j
```

Run (defaults to the PR62 asset paths; or pass `[model.pt] [golden_io.json]`):

```bash
/root/Nuka-Physics/tools/inference/build/go2_policy_check
# explicit paths:
/root/Nuka-Physics/tools/inference/build/go2_policy_check \
    /root/third_party/go2_pr62/motion.pt \
    /root/third_party/go2_pr62/golden_io.json
```

### Runtime library path

The executable bakes the torch `lib/` dir into its RPATH
(`BUILD_RPATH = <torch>/lib`), and libtorch's CUDA dependencies resolve via
libtorch's own rpath — so **no `LD_LIBRARY_PATH` is required** in this
environment.

If you ever relocate the binary or the torch package and the loader cannot find
a lib, set:

```bash
export LD_LIBRARY_PATH=$TORCH_CMAKE/../../../lib:$LD_LIBRARY_PATH
# i.e. .../site-packages/torch/lib
```

We run inference on **CPU** (`torch::kCPU`); no GPU is needed.

## Expected output

```
SHA-256 match : YES

case                    in_dim    out_dim   max_err       result
----------------------------------------------------------------
zeros                   48        12        0.000e+00     PASS
ones                    48        12        0.000e+00     PASS
randn_seed0             48        12        0.000e+00     PASS
default_obs_cmd0.5      48        12        0.000e+00     PASS
----------------------------------------------------------------
overall max_err = 0.000e+00  (tol 1.000e-04)

OVERALL: PASS  (golden_match=yes, sha256_match=yes)
```

## Notes / build environment

- Toolchain: `g++-10` (10.5.0), CMake 3.30, libtorch `2.6.0+cu124`.
- The harness does **not** vendor the `.pt`; it reads it by path.
- A negative-control run (a deliberately corrupted golden) makes the harness
  print `FAIL` and exit `1`, confirming the check is real rather than a no-op.
