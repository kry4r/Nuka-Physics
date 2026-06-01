---
name: Bug report
about: Report a problem in Nuka Physics
title: "[bug] "
labels: bug
assignees: ''
---

## Describe the bug

<!-- A clear and concise description of what the bug is. -->

## Reproduction

<!-- Minimal steps or a minimal script/scene that reproduces the problem. -->

1.
2.
3.

```text
# minimal repro (command / Python snippet / scene file excerpt)
```

## Expected behavior

<!-- What you expected to happen. -->

## Actual behavior

<!-- What actually happened (error messages, stack traces, wrong values). -->

## Environment

- GPU model (e.g. RTX 5080):
- CUDA version (e.g. 12.8):
- OS / kernel:
- Compiler (e.g. g++-10):
- Nuka version / commit:
- Python version and torch/jax versions (if using the bindings):

## Determinism-relevant info

<!-- Determinism is a core contract. Please fill these in if relevant. -->

- Determinism mode: [ ] D1 (strong, bit-exact, default)  [ ] D2 (weak, opt-in)
- Is the result **reproducible bit-for-bit across two runs**? [ ] yes [ ] no [ ] N/A
- Number of environments / batch size:
- Are you using the CUDA-graph step path (`StepGraph`)? [ ] yes [ ] no

## Additional context

<!-- Anything else that helps us understand the issue. -->
