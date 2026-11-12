# Основания для ответов DEFlow

Исторический аудит 6–7 ноября 2026. Числа, версии окружения и упоминания «текущего кода» ниже относятся к тому аудиту и исходной статье, а не к новой кампании E01. Первичный источник — [предоставленный PDF](../sources/DEFlow_main.pdf). В рамках этого аудита новые benchmark trainings не проводились.

## Результаты и статистика

| Утверждение | Что подтверждено | Ограничение |
|---|---|---|
| Docking | Средние и SD Table 1 воспроизводятся из сохранённых результатов. Среднее четырёх относительных приростов к DIME — **10,0902%**. | U4 — successful coverage, не процент успешных episodes. Высокий разброс на tight-вариантах. |
| Matched explicit | Все 12 исходных docking-пар имеют одинаковые actor shapes и saved args, кроме integration и имён запуска. | Меняется sampling вместе с differentiation. Это не joint-vs-sequential ablation. |
| Seeds | Wide-tight explicit агрегирует **два завершённых run**. Depth-иллюстрация использует одну пару. | Средние completed runs не включают численно упавшие runs. Display seed не всегда равен исходному RNG seed. |
| Manipulation | Лучшее mean AUC-return на 4/5 задач. Это нормированный trapezoidal AUC по recorded steps. | Не wall-clock и не простое среднее checkpoint values. Удаление ранних интервалов сохраняет 4/5 на окнах 100k–500k и 250k–500k, но не устраняет различия recipes. |
| Final success | Peg: DEFlow 39,3%, SAC 72,3%. PickCube: DEFlow 87,3%, DIME 93,0%. Button/PushWall DEFlow: 8,7%/0,3%. | Это Table 5. Table 4 — отдельное paired reevaluation, где original PickCube/Peg равны 87,0%/38,0%. |
| MuJoCo | Table 6 воспроизводится. Consistent superiority DEFlow не показано. | Сумма ненормированных rewards выше у SAC, но зависит от масштабов задач. |

Подробные пересчёты, input hashes и попытки опровержения сохранены в архивных отчётах `05_metrics.md`, `09_rebuttal_strategy.md`, `10_adversarial_crosscheck.md`.

## Численный аудит статьи

Table 3 сравнивает с численно найденным equilibrium reference. В коде comparator используется sequential FP64 SciPy root solve, guarded density objective и unshifted adjoint. Это не проверка производной или глобального likelihood фактического finite solver.

| Параметр | Подтверждённые данные |
|---|---|
| Размеры выборок | Action/density: B1, 384 inputs на task/arm, PushWall 381. Gradients: B8, 48 batches, PushWall 46. Saved training batch — B1024. |
| Четыре задачи кроме PushWall | Original P95 gradient error ≤0,836%, guarded density error ≤0,00191 nat. Tighter P95 gradient error <0,002%. Это не верхняя граница всех ошибок. |
| PushWall | Gradient median 13,7% → 2,84%, P95 96,1% → 50,3%. P99/max ухудшаются с 128,3%/144,7% до 241,1%/349,9%. |
| Reference exclusions | Три PushWall inputs не получили валидированный reference. Два затронутых B8 batches исключены полностью. |
| Guards | Во всех 3 840 сохранённых B1 строках determinant clipping и tanh floor не активны. Whole-logp clipping и B8 adjoint conditioning эти записи не устанавливают. |
| Cap hits | Средние evaluation cap-hit rates 47,19% → 13,14%. Это не частота training non-convergence. Cap-hit и tolerance failure различаются. |
| Коэффициент penalty | Полный `lam_jac` корректен для audited final checkpoints на 500k. При early-checkpoint reuse нужно учитывать warm-up. |

Код comparator: [audit_manipulation_solver_reeval.py](/Users/ruslanageev/dev/DEFlow/src/analysis/audit_manipulation_solver_reeval.py:54), [reference_root](/Users/ruslanageev/dev/DEFlow/src/analysis/manipulation_numerical_verification.py:79). Полный разбор — архивный `07_numerical_audit.md`.

## Текущий код и training recipes

Проверенный commit: `f81ae7a75107eb15608266054e491faa0f7d85c9`. **Точные historical engines локально не найдены.** Совместимость weights или совпадение современных копий не доказывают, что эта же программа породила paper results.

| Находка | Доказательство | Что из неё не следует |
|---|---|---|
| Без градиентов actor возвращает solver output `q`, с градиентами — `g(q)`. | [Attachment](/Users/ruslanageev/dev/DEFlow/src/sac_anderson_flow.py:432), CPU-пробы всех четырёх копий. | Не измерен масштаб на исторических trained policies. |
| Parameter derivative берётся в `q`, adjoint — в `r=g(q)`. | [Dense backward](/Users/ruslanageev/dev/DEFlow/src/sac_anderson_flow.py:467), scalar counterexample и exact-root controls. | Не опровергнута equilibrium-теорема. Пропущенных direct entropy/penalty gradients и block-transpose ошибки не найдено. |
| Stopping зависит от среднего residual по batch. | [Anderson](/Users/ruslanageev/dev/DEFlow/src/sac_anderson_flow.py:254), synthetic batch-composition check. | B1/B8 не заменяют B1024 validation. Cap-hit не обязательно означает tolerance failure. |
| Logp clipping, если включён, достигает actor, critic target и temperature. | [Target path](/Users/ruslanageev/dev/DEFlow/src/sac_anderson_flow.py:1221), [actor/alpha path](/Users/ruslanageev/dev/DEFlow/src/sac_anderson_flow.py:1268). | Не доказано, что guard counterexamples объясняют ошибки Table 3. |
| Entropy tracking перезаписывает scheduled target. | [Alpha update](/Users/ruslanageev/dev/DEFlow/src/sac_anderson_flow.py:1290). Saved PushWall recipe включает tracking. | Нельзя автоматически утверждать, что stored schedule реально использовался историческим engine. |
| Warm-up DEFlow 4k против 50k у SAC/SAC-Flow/DIME. Также различаются batch/UTD/LR/tau. | [Recipes](/Users/ruslanageev/dev/DEFlow/src/repro/run_manipulation_stage1.py:28), все 60 manipulation saved configs. | Ранний return не изолирует implicit sampling. Matched docking explicit разделяет настройки DEFlow и этим confound не затронут. |
| Current loop даёт 23 004 против 8 update iterations к первой оценке на 50k. | [Update loop](/Users/ruslanageev/dev/DEFlow/src/sac_anderson_flow.py:1214), независимый арифметический пересчёт. | Это не восстановленные historical optimizer counters и не сумма всех отдельных `optimizer.step` calls. |
| Общий trainer считает recorded slots, включая reset-only calls. Docking считает реальные transitions. | [General loop](/Users/ruslanageev/dev/DEFlow/src/sac_anderson_flow.py:1197), [docking loop](/Users/ruslanageev/dev/DEFlow/src/docking_full/sac_anderson_flow.py:1200). | Нельзя переносить ограничение общего trainer на основной docking experiment. |

## Что осталось от исторических запусков

- Все 60 manipulation event exports содержат только две evaluation series, все 1 200 scalar timestamps равны нулю. Equal-time learning curves из них не восстановить.
- Проверенные DEFlow checkpoints содержат actor, online critics, temperature, args и step. Полный replay/target critics/optimizer/RNG state отсутствует. Восстановление weights не равно faithful resume.
- Display labels DEFlow PickCube/PushCube 1/2/3 соответствуют исходным seeds 2/4/5. Это разные seeds, но не paired seeds с baselines. Причины всех исторических исключений независимо не подтверждены.
- Sequential reference уже есть, но общего hardware/accuracy performance comparison с joint solver нет. `n_iter` не включает все вычисления density/Jacobians/backward.

Происхождение: архивные `04_protocol.md`, `06_provenance.md`, `08_runtime_tests.md` и [seed mapping](/Users/ruslanageev/dev/DEFlow/docs/checkpoint_seed_changes_2026-09-29.json:5).

## Проверки и воспроизведение

**102 CPU-теста прошли**, девять training-loop tests исключены. Artifact validator прошёл. Torch 2.5.1 и NumPy 1.26.4 совпадают с requirements. SciPy 1.10.1 вместо 1.12.0 и Matplotlib 3.4.3 вне заявленного диапазона ограничивают утверждение о parity окружения. Тесты не проверяют историческую learning robustness.

[Архив воспроизводимого аудита](code_audit_2026-11-07.tar.gz) содержит agent reports, scripts, JSON evidence, test receipts, manifest и прежние сводки на момент реорганизации. Старые пути внутри него — исторические. Для повторного запуска scripts нужно восстановить `2026-11-06` внутри `rebuttal`, как в архиве. Общие задачи ведутся в [ISSUES.md](../ISSUES.md), действующий протокол — в [E01 README](../../experiments/E01-solver-comparison/README.md).
