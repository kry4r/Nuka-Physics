# Security Policy

## Reporting a vulnerability

If you discover a security vulnerability in Nuka Physics, please report it
**privately**. Do **not** open a public GitHub issue, pull request, or
discussion for security reports.

Email: **igloo-group@pipeline.sbcc.edu**

Please include, where possible:

- A description of the vulnerability and its potential impact.
- Steps to reproduce (a minimal proof of concept is ideal).
- The affected version/commit and your environment (GPU model, CUDA version,
  OS).
- Any suggested mitigation, if you have one.

## Response process

- We aim to **acknowledge** your report within **5 business days**.
- We aim to provide an **initial assessment** (whether it is accepted as a
  vulnerability, and a rough severity) within **10 business days**.
- We will keep you informed of remediation progress and coordinate a disclosure
  timeline with you. Please give us reasonable time to investigate and release a
  fix before any public disclosure.

These windows are targets for a research-grade project maintained by a small
team and are not contractual guarantees.

## Supported versions

| Version | Supported          |
| ------- | ------------------ |
| 0.5.x   | :white_check_mark: |
| < 0.5   | :x:                |

Security fixes are provided for the **v0.5.x** line. Older releases are not
supported.

## Scope and disclaimer

Nuka Physics is **research-grade simulation software**. It is intended for
robotics simulation, reinforcement learning, and research use, and is provided
on an "AS IS" basis under the [Apache License 2.0](LICENSE) with no warranty. It
is not hardened for, and should not be relied upon as a security boundary in,
adversarial or safety-critical production deployments. Inputs such as scene
files (MJCF/URDF/USDA) and model assets should be treated as trusted; do not
load untrusted scene/model files from unknown sources.
