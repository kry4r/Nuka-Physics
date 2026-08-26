# Go2 Skill Models

These TorchScript artifacts are the project-local deployment names used by
`examples/demo/go2_skill_infer.py`:

- `nuka_go2_backflip.pt`: double-backflip policy replayed through Nuka PD control.
- `nuka_go2_front_handstand.pt`: front-handstand policy replayed through Nuka PD control.

The replay harness documents the Nuka-side observation, action, PD, timing, and
trajectory contracts.