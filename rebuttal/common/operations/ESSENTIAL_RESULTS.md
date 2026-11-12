> **Current publication status, 12 November 2026:** E01 is stopped, E02 and E03 are complete. The later [restart authorization](RESTART_E04_E05_AUTHORIZATION-20261112.json) supersedes the previous global STOP only for six E04/E05 `restart1` jobs. All six were verified training in this publication snapshot. Earlier stop/status text below is historical. Start with [the evidence index](../../EVIDENCE_INDEX.md).

# Главные результаты DEFlow — локальная копия перед отключением GPU

Все раны E01–E05 остановлены пользователем 12 ноября 2026. Координаторы и службы очередей/синхронизации остановлены. Не запускать повторно обучение, evaluation, audits или службы без нового запроса пользователя. Исходные результаты не удалялись.

Сохранены **13 финальных stochastic coverage evaluations по 1000 эпизодов**, **18 завершённых численных audits**, learning curves, manifests/precision evidence, периодические и финальные checkpoints, failed/stopped receipts и логи. Данные доступны локально для анализа без GPU-серверов.

| Эксперимент | Сохранённое состояние | Локальный каталог |
| --- | --- | --- |
| E01 solver comparison | 10 final evaluations, 9 audits, N2 scientific failure, N1 user stop около164k. Последний N1 checkpoint150020. | [mirror](../../experiments/E01-solver-comparison/runs/deflow-campaign-20261111/revision3/mirror/) |
| E02 sequential Anderson | 3 final evaluations и final checkpoints, закончены до остановки. | [mirror](../../experiments/E02-sequential-anderson/runs/sa-20261112/mirror/) |
| E03 tighter-setting audit | 9 audits, все raw.pt, summaries, masks и проверенные receipts. | [mirror](../../experiments/E03-tighter-anderson-audit/runs/tol1e-6-cap100-20261112/mirror/) |
| E04 joint cap100 | 3 user-stopped runs около26–30k, checkpoints около25k, events, exits -15. Финальных evaluations нет. | [mirror](../../experiments/E04-joint-anderson-cap100/runs/a100-20261112/mirror/) |
| E05 joint cap10 | 3 user-stopped runs около5k, events, exits -15. Production checkpoints ещё не создавались. Сохранены 3 последние capsules с weights/inputs, отдельно проверены SHA256. Финальных evaluations нет. | [mirror](../../experiments/E05-joint-anderson-cap10/runs/a10-20261112/mirror/) |

## Начать анализ

- [Итоговые learning results CSV](shutdown-20261112T074015Z/learning-results.csv) и [JSON](shutdown-20261112T074015Z/learning-results.json) содержат все 21 run IDs, состояния, метрики 13 финальных evaluations и пути к checkpoints. STOPPED_BY_USER не является научным non-finite failure. Только N2 имеет отдельный scientific failure.
- [E01 отчёт по девяти audits](../../experiments/E01-solver-comparison/runs/deflow-campaign-20261111/revision3/mirror/ops/idle-audits-v1/analysis/REPORT.md), рядом metrics.json и per_audit.csv.
- [E03 tighter-setting report](../../experiments/E03-tighter-anderson-audit/analysis/REPORT.md), рядом metrics.json.
- [Разбор Newton](../../experiments/E01-solver-comparison/analysis/newton-investigation-20261112/README.md).
- В каждом mirror: `results/<run_id>/runs/*/events.out.tfevents.*` — learning curves, `ckpt_step_*.pt` — checkpoints. `evaluations/<run_id>/evaluation.json` — final metrics, episode records, eval seed и checkpoint hash. Intermediate deterministic training eval хранится отдельно в events.
- E01/E03: `audits/<audit_id>/{raw.pt,summary.json}` содержат tensors, reference validity и failure masks. Все выбранные E01 input capsules также доступны в E01 mirror.

## Проверка и ограничения

[Preservation receipt](shutdown-20261112T074015Z/ESSENTIALS_VERIFIED.json) и per-campaign `*-essential-verified.json` подтверждают совпадение SHA256 и размеров **1064 файлов, 1491261394 байт** с серверами после остановки. Отдельно проверены 72 E03 raw/receipt files и SHA256 обоих E03 reports. Exact per-file hashes сохранены в той же папке. Три E05 capsules включены в проверенный E05 набор.

Для E01/E03 full-batch reference comparisons допустимы только на пяти полностью валидных batches, 5120 trajectories. Четыре остальных batches имеют reference-invalid rows, их N/A comparisons и masks сохраняются. Timing есть для всех девяти cases. E03 tighter-setting audit использует A25-trained capsules, а не T-trained policy capsules.

E01 использует RTX4090/torch2.5.1+cu124, E02/E04/E05 — RTX5090/torch2.7.1+cu128. Не делить elapsed разных hardware/shared-GPU intervals и не называть это чистым solver speedup. Финальные оценки имеют eval_seed26152. Неполные раны не добавлять к трёхseedовым final aggregates.

Полные solver_calls.jsonl, все training capsules, venv и отменённые полные архивы не переносились. Основные табличные результаты, learning curves, выбранные audit inputs и raw audits сохранены. Отмена полного raw-переноса остаётся в силе.
