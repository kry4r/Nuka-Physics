<!--
Thanks for contributing to Nuka Physics!
Please read CONTRIBUTING.md and complete the checklist below.
-->

## Summary

<!-- What does this PR change, and why? Link any related issue or master-plan item. -->

## Checklist

- [ ] **Physics-smell lint passes** — `python tools/lint/physics_smell.py` is clean (no new allowlist entries to land physics code).
- [ ] **C++ tests pass** — `ctest --test-dir build-cuda128 --output-on-failure`. Only the **2 known pre-existing oracle fails** may fail (`V01FoundationE2E.Phase6CudaAbaMatchesGo2AndH1MuJoCoOracle` #38 and `FeatherstoneOracle.RandomSampleGoldensMatchCudaAba` #206); no new failures.
- [ ] **Python tests pass** — `pytest python/tests/`.
- [ ] **Determinism preserved** — no new floating-point atomics (e.g. float `atomicAdd`) in physics code paths; D1 strong determinism (bit-exact two-run) is not regressed.
- [ ] **No edits to generated code** — `src/codegen/generated/**` is untouched (regenerate via the build instead), and no protected golden artifacts were changed without a reviewed protected-change proposal.
- [ ] **CLA signed** — the CLA-assistant bot check is green for this PR's author.

## Notes for reviewers

<!-- Anything reviewers should pay special attention to (perf, determinism, ABI, codegen). -->
