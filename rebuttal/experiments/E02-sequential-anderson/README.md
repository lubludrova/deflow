# E02 — sequential Anderson на docking

Запрошен пользователем 12 ноября 2026: реализовать недостающий sequential Anderson baseline, проверить на сохранённых данных и запустить три matched seed на сервере с портом 13534. Это отдельное расширение E01. Замороженная кампания E01 и опубликованные results не меняются.

## Научный вопрос и ограничения

Связанные замечания: D-C01/D-C02/D-C04 (joint versus sequential), D-C13 и численные вопросы VrBX. Исходный разбор: historical Newton investigation (preserved in the local archive).

Sequential Anderson сохраняет исходный Anderson kernel и решает по очереди четыре implicit Euler equations. Каждое уравнение стартует от предыдущего найденного решения, с новой историей. Точно сохранены m=5, lam=1e-2, beta=0.7, native tolerance=1e-5, cap=25 **на каждый step**, общий mixed-point attachment, shifted dense backward, density и Jacobian penalty. Никаких fallback на Newton или изменений convergence criterion.

Это сравнение организации forward внутри одного семейства solver, а не равного compute. У joint cap25 относится к совместной системе, у sequential к каждому из четырёх шагов. Native stopping использует batch mean. Common fresh local residual проверяется отдельно: atol=1e-7 + rtol=1e-5 × max(norm(q), norm(previous)). Cap hit, native mean convergence и fresh per-sample failure сохраняются раздельно. Общий backward сохраняет контроль E01, но не сравнивает оптимизированный sequential backward.

Seeds 1/2/3, MultiGoalDockTight-v0, 200000 реальных переходов, num_envs=8, batch=1024, learning_starts=4000, utd_ratio=0.5. Архитектура, оптимизаторы, reward, environment, evaluation и остальные параметры совпадают с E01. Проверка полного configuration dictionary разрешает только seed и имя run. Все девять файлов engine имеют исходные E01 hashes.

Новый сервер имеет RTX 5090. Используется strict FP32, PyTorch 2.7.1+cu128, вместо E01 2.5.1+cu124. Это необходимая адаптация к Blackwell, поддержка которого появилась в [PyTorch 2.7](https://pytorch.org/blog/pytorch-2-7/). Поэтому это matched scientific settings/seeds, но не matched hardware/runtime и не bitwise reproduction. Историческое shared-GPU elapsed E01 нельзя делить на elapsed E02 и называть чистым solver speedup.

## Организация

- [code/package/deflow_rebuttal/sequential_anderson.py](code/package/deflow_rebuttal/sequential_anderson.py): новый forward.
- [code/package/deflow_rebuttal/run_docking.py](code/package/deflow_rebuttal/run_docking.py): dispatch, telemetry, capsules.
- [code/validate_capsules.py](code/validate_capsules.py): одни и те же полные A2/N3 capsules для joint Anderson, Newton и sequential Anderson.
- [runs/sa-20261112/ops/manifest.json](runs/sa-20261112/ops/manifest.json): точная очередь, команды и frozen hashes.
- [runs/sa-20261112/ops/profile.json](runs/sa-20261112/ops/profile.json): доступ и пути без credentials.
- [analysis/validation.md](analysis/validation.md): проверки и их границы.
- [runs/sa-20261112/mirror](runs/sa-20261112/mirror): компактное зеркало сервера.

Remote root: `/home/featurize/deflow-sequential-anderson-20261112` на `workspace.featurize.cn:13534`.

Очередь: GPU0 → sequential_anderson_s1 → evaluation1000. GPU1 → sequential_anderson_s2 → evaluation1000. GPU2 → sequential_anderson_s3 → evaluation1000. Затем verified final sync. Seed evaluation=26152. Final coverage evaluation стохастическая, как в E01, intermediate training eval — deterministic. Дополнительных обучений Newton/Broyden/joint Newton и новых девяти audits в этой очереди нет.

Один training/evaluation process на физическую GPU. Ноль retries и checkpoint resume. Перед стартом — 60 секунд проверки свободных GPU. tmux supervisors переживают SSH disconnect, сохраняют PID/start-time identity, реальные exit codes и проверяют checkpoint/horizon. Оценка начинается только после успешного обучения и проверки локальной копии итогового checkpoint. Ошибка сохраняется, автоматической смены конфигурации нет.

## Проверить состояние

Из корня DEFlow:

```bash
python3 rebuttal/experiments/E02-sequential-anderson/runs/sa-20261112/ops/status_verified.py
python3 rebuttal/experiments/E02-sequential-anderson/runs/sa-20261112/ops/sync_local.py
```

Sync через отдельную launchd-службу каждые 120 секунд. Копируются immutable checkpoints, events, summaries, receipts, validation. Полные solver_calls и training capsules остаются на сервере, compact mirror не является полным raw backup. Удалений нет. ACK проверяет SHA256 checkpoint файлов. Для автоматической оценки нужен свежий ACK.

SSH control connection хранит только существующую авторизованную сессию. Пароль в проект не записан. После потери транспорта потребуется заново открыть control connection интерактивно. Обучения продолжатся, оценки останутся ждать verified sync. Старые E01 sync и keepawake не меняются.

Production commands берутся из manifest. Не повторять `runner.py --start` и не запускать jobs вручную поверх tmux. Повторный submit и попытка повторного run_id отвергаются атомарными claims.

## Подтверждённый запуск

**Итог, 12 ноября 2026:** все три seed завершили 200000+ переходов и финальную стохастическую оценку по 1000 эпизодов. U4 = **0.6197 ± 0.0568**, success = **92.13 ± 5.41%** (mean ± sample SD по seeds). Финальные checkpoints и оценки сохранены локально, SHA256 проверены. [Итоговый отчёт](analysis/final-results-20261112/report.md) и [проверка файлов и метрик](analysis/final-results-20261112/verification.json). Полные raw traces/capsules остаются на сервере согласно compact-mirror policy.

12 ноября 2026 в 10:06 Asia/Shanghai все три runs находятся в TRAINING: S1 7021, S2 7025, S3 7024 перехода. Это snapshot, не live status. [LAUNCH_VERIFIED.json](runs/sa-20261112/LAUNCH_VERIFIED.json) содержит проверенные PID/GPU и результаты gates. Финальные learning/evaluation результаты ещё не получены. Для seed2 добавлен отдельный read-only identity overlay, описанный в [HANDOFF.md](runs/sa-20261112/HANDOFF.md), без изменения frozen science или supervisor.
