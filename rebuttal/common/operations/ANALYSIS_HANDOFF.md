> **Current publication status, 12 November 2026:** E01 is stopped, E02 and E03 are complete. The later [restart authorization](RESTART_E04_E05_AUTHORIZATION-20261112.json) supersedes the previous global STOP only for six E04/E05 `restart1` jobs. All six were verified training in this publication snapshot. Earlier stop/status text below is historical. Start with [the evidence index](../../EVIDENCE_INDEX.md).

> **Все раны E01–E05 остановлены пользователем 12 ноября 2026. Главные результаты сохранены и проверены на Mac. Начать с [ESSENTIAL_RESULTS.md](ESSENTIAL_RESULTS.md). Серверы больше не нужны для анализа этих сохранённых результатов. Новые запуски запрещены без нового запроса.**

# Данные для анализа E01 и E02

Обновлено 12 ноября 2026, 14:19 Asia/Shanghai. Пользователь отменил полный перенос raw на Mac. Анализировать полные данные на серверах. Все исходные результаты сохранены. Новые полные скачивания и упаковку не возобновлять без запроса пользователя. Существующая компактная синхронизация и обучение продолжаются.

Перед работой прочитать `/Users/ruslanageev/dev/DEFLOW/rebuttal/AGENTS.md` и `/Users/ruslanageev/dev/DEFLOW/rebuttal/RUNNER_HANDOFF.md`. Ниже указаны независимые кампании. Не смешивать их, историческую revision2 с TF32 и опубликованные cohorts.

## Серверы и корневые пути

| Кампания | SSH | Полный набор данных на сервере |
| --- | --- | --- |
| E01 strict-FP32 revision3 | `featurize@workspace.featurize.cn`, порт `18290` | `/home/featurize/deflow-campaign-20261111/revision3` |
| E02 sequential Anderson | `featurize@workspace.featurize.cn`, порт `13534` | `/home/featurize/deflow-sequential-anderson-20261112` |

Доступ с Mac к E01 через существующий wrapper:

```bash
/Users/ruslanageev/dev/DEFLOW/rebuttal/experiments/E01-solver-comparison/runs/deflow-campaign-20261111/revision3/ops/ssh.sh featurize@workspace.featurize.cn
```

Доступ с Mac к E02 через существующую авторизованную SSH control connection:

```bash
ssh -S /tmp/deflow-sa-20261112.sock -o BatchMode=yes -p 13534 featurize@workspace.featurize.cn
```

Credentials в документах отсутствуют. После потери авторизованной E02 connection может потребоваться повторная аутентификация. Не запускать повторно coordinator/setup или training ради доступа. Python каждой кампании находится в `<remote_root>/venv/bin/python`.

## E01: обучение, evaluation и исходная телеметрия

Все пути этого раздела относительно `/home/featurize/deflow-campaign-20261111/revision3`.

| Run IDs, каждый с `s1`, `s2`, `s3` | Условие |
| --- | --- |
| `newton_s1`, `newton_s2`, `newton_s3` | Текущее Newton implementation, исходные настройки frozen contract |
| `anderson_s1`, `anderson_s2`, `anderson_s3` | Joint Anderson, tol 1e-5, cap 25 |
| `anderson_highcap_s1`, `anderson_highcap_s2`, `anderson_highcap_s3` | Joint Anderson, tol 1e-5, cap 500 |
| `anderson_tighter_s1`, `anderson_tighter_s2`, `anderson_tighter_s3` | Joint Anderson, tol 1e-6, cap 100 |

Для каждого `<run_id>`:

- `results/<run_id>/solver_calls.jsonl` — полная per-call телеметрия, native/fresh residuals, роли вызовов и actual global_step.
- `results/<run_id>/telemetry_summary.json` — сводка телеметрии.
- `results/<run_id>/capsule_*.pt` — полные сохранённые actor/critic/loss inputs для воспроизведения. Число в имени — счётчик вызовов solver, а не число environment transitions. Для сопоставления использовать записанный `global_step`.
- `results/<run_id>/runs/<run_name>/events.out.tfevents.*` — кривые train return, deterministic intermediate eval, losses и steps.
- `results/<run_id>/runs/<run_name>/ckpt_step_*.pt` — чекпойнты. Точный итоговый checkpoint и global_step перечислены в `results/<run_id>/DONE.json`. `ckpt.pt` активного обучения может обновляться.
- `results/<run_id>/{manifest.json,actor_kwargs.json,runtime_precision.json}` — параметры, hashes и precision evidence.
- `results/<run_id>/{DONE.json,FAILED.json}` — фактические training outcomes, включая сохранённый traceback.
- `evaluations/<run_id>/evaluation.json` — финальные stochastic coverage оценки: 1000 episodes, eval_seed 26152, U4, success, mean return и checkpoint hash. Не подменять ими deterministic training eval или наоборот.
- `ops/idle-audits-v1/campaign_report.json` и `.md` — действующая сводка. Ранние reports старых coordinators могут содержать административные terminal labels.
- `ops/execution_contract.json` — frozen science/источники. SHA256: `300614d6c0abfbb1bbd5f722effbc5c00774f7213c7598dc68b6527c0225c0ea`.

Последняя проверка **14:16 Asia/Shanghai**: 10 runs имеют принятый финальный evaluation. `newton_s1` ещё обучается, step 137605, GPU0. `newton_s2` завершился ошибкой на step 131908: non-finite pi_loss. Повторов и resume нет. Метрики N1 и E02 ниже являются датированными snapshots. Перед выводами об их текущем состоянии обновить status.

## E01: девять paired audits и готовая сводка

Все девять audits завершены и уже проверены в компактной локальной копии. На сервере:

- `audits/audit_a{1,2,3}_{25000,100000,200000}/summary.json` — численные результаты, actual capsule/source hashes, precision и ограничения.
- В тех же каталогах `raw.pt` — исходные tensors, reference-invalid masks и результаты каждого solver.
- `ops/idle-audits-v1/analysis/REPORT.md` — готовый разбор.
- `ops/idle-audits-v1/analysis/metrics.json` — агрегированные численные показатели.
- `ops/idle-audits-v1/analysis/per_audit.csv` — таблица 36 строк, 9 cases × 4 solver conditions.
- `ops/idle-audits-v1/{manifest.json,PROTOCOL.md}` — выбранные actual steps, hashes, исходный запрос и границы протокола.

Аудиты сравнивают A25/A100/A500 при tol1e-5 и текущий Newton на одинаковых A-policy capsules. Исходный полный SAC actor objective сохранён. Timing есть для всех 9 cases. Полные reference-gradient/loss/density comparisons допустимы только для 5 cases, 5120 trajectories. Все три cases около 100k и A3 около 200k имеют reference-invalid rows, соответственно 2/1/1/1. Их full-batch comparisons остаются N/A. Маски и строки сохранять, не превращать их в утверждение о полностью валидном reference.

Не обобщать эти capsules на всю траекторию обучения, на N-policy или на причину падения N2. Не выдавать equilibrium reference за exact finite-program likelihood. Repeats timing не являются независимыми training seeds.

## E02: sequential Anderson ×3

Все пути раздела относительно `/home/featurize/deflow-sequential-anderson-20261112`.

- Run IDs: `sequential_anderson_s1`, `sequential_anderson_s2`, `sequential_anderson_s3`, физические GPU0/1/2.
- `results/<run_id>/` имеет те же основные solver_calls, capsule, TensorBoard, checkpoint и manifest artifacts, что E01.
- `attempts/<run_id>/0000/` — training/evaluation logs, process identities, реальные exit receipts и `{TRAINING,EVALUATION}_ACCEPTED.json` по мере завершения. Директория может также содержать FAILED.json.
- `evaluations/<run_id>/evaluation.json` появится после принятого обучения и verified checkpoint sync. Протокол: 1000 stochastic coverage episodes, eval_seed 26152.
- `ops/manifest.json` — frozen sources/configs, команды и acceptance gates. SHA256: `31f4c2449287141d788de2f9e00ac732b335e64676bf668ad8247fe8c4b932f5`.
- `validation/` — smoke, pilot, fixed-capsule diagnostics и сохранённые ранние failed diagnostic attempts.
- `code/package/deflow_rebuttal/sequential_anderson.py` — forward implementation.

Проверка **14:16 Asia/Shanghai**: S1 step180541, S2 step181500, S3 step180583, все TRAINING. На тот момент ETA до конца обучения около 27–30 минут, final evaluation отдельный этап.

Условие: strict FP32, native Anderson per implicit step, cap25/tol1e-5, fresh history для каждого из четырёх steps. Density/backward сохранены из E01. Cap25 sequential относится к каждому step, joint cap25 — к совместной системе. Hardware/runtime различаются: E01 RTX4090, torch2.5.1+cu124, E02 RTX5090, torch2.7.1+cu128. Не трактовать отношение elapsed этих campaigns как чистый solver speedup.

## Что уже есть на Mac

Компактное E01 зеркало:
`/Users/ruslanageev/dev/DEFLOW/rebuttal/experiments/E01-solver-comparison/runs/deflow-campaign-20261111/revision3/mirror`

Компактное E02 зеркало:
`/Users/ruslanageev/dev/DEFLOW/rebuttal/experiments/E02-sequential-anderson/runs/sa-20261112/mirror`

Это компактные копии. Для полного solver_calls и всех training capsules использовать серверы. E01 audit raw.pt, выбранные audit capsules и сводные таблицы уже скопированы и SHA-256 проверены.

Готовый отдельный разбор Newton и matched-source evidence находится здесь:
`/Users/ruslanageev/dev/DEFLOW/rebuttal/experiments/E01-solver-comparison/analysis/newton-investigation-20261112/README.md`

Связанные научные документы: E01/E02 README, `rebuttal/common/ISSUES.md`, reviewer `rebuttal/reviewers/dkt6.md` и `VrBX.md`. Новые analysis artifacts сохранять под `rebuttal/experiments/E01-solver-comparison/analysis/` или соответствующим `E02.../analysis/`.

## Обновить состояние без изменения запусков

С Mac, из `/Users/ruslanageev/dev/DEFLOW`:

```bash
python3 rebuttal/common/operations/status.py --json
python3 rebuttal/experiments/E02-sequential-anderson/runs/sa-20261112/ops/status_verified.py
```

Не запускать E02 frozen `status.py` вместо corrected `status_verified.py`. Последний учитывает отдельно подтверждённую пустую cmdline при первоначальном чтении PID S2. Анализ не требует training restart, изменения frozen configs, новых audits или GPU diagnostics поверх текущих runs.

## Отменённая полная копия

Полный перенос отменён пользователем 12 ноября. Оба download.py и их transport children остановлены. E01 capture/supervisor остановлены с проверенными identities и реальным EXIT -15. E02 capture успел завершиться раньше, его tmux session уже отсутствует. Сохранённые архивные фрагменты не являются локальной проверенной полной копией и не должны запускаться заново.

Служебные серверные остатки, не основной источник научного анализа:

- E01: `/home/featurize/deflow-campaign-20261111/revision3/preservation/full-results-20261112T055114Z/` — прерванная упаковка.
- E02: `/home/featurize/deflow-preservation/deflow-sequential-anderson-20261112/full-results-20261112T060123Z/` — завершённый dated archive, перенос на Mac отменён.

Локальные cancellation/stop receipts лежат в `full-backups/<timestamp>/` соответствующей кампании. Ничего не удалялось. Обычная E02 compact sync восстановлена, fresh ACK проверен после отмены полного переноса.
