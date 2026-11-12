# Current handoff — 12 November 2026

E01 is stopped. E02 sequential Anderson has three completed 200k trainings and
three accepted 1000-episode evaluations. E03 has nine completed numerical audits.
The only active training campaigns are E04/E05 `restart1`, three seeds each,
explicitly reauthorized after the global stop. No other restart is authorized.

Current immutable evidence and publication limits: [EVIDENCE_INDEX.md](EVIDENCE_INDEX.md).
Responses: [reviews](reviews/). Shared notes and all 35 concerns: [common/ISSUES.md](common/ISSUES.md).

| Active campaign | Frozen handoff | Status command from repository root |
| --- | --- | --- |
| E04 cap100 | [handoff](experiments/E04-joint-anderson-cap100/runs/a100-20261112-restart1/HANDOFF.md) | `python3 rebuttal/experiments/E04-joint-anderson-cap100/runs/a100-20261112-restart1/ops/status.py` |
| E05 cap10 | [handoff](experiments/E05-joint-anderson-cap10/runs/a10-20261112-restart1/HANDOFF.md) | `python3 rebuttal/experiments/E05-joint-anderson-cap10/runs/a10-20261112-restart1/ops/status.py` |

The dated publication snapshots are fixed. Live mirrors and synchronization
continue separately at their existing local and remote paths. Do not change
scientific settings, restart original campaigns, or bypass the E04 final-ACK /
E05 evaluation gates. Publication did not stop jobs or alter launch services.

The ignored local archive holds old attempts, diagnostics and checkpoints.
Ignored compatibility symlinks preserve historical absolute/local paths.
Do not remove those aliases while any frozen command or sync still uses them.
