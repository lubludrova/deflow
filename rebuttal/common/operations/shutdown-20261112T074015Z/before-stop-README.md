# Состояние и синхронизация

Из корня `~/dev/DEFLOW`:

```bash
python3 rebuttal/common/operations/status.py
python3 rebuttal/common/operations/status.py --json
```

Обе команды только читают состояние. Проверяются frozen hashes и process identities, выводятся шаги, recent SPS, terminal states, alert markers и свежесть sync. ETA относится к текущему обучению, без последующих оценок и задач.

## Источники

- [active_campaign.json](active_campaign.json) — канонический профиль и реальные локальные/серверные пути. `.agents/remote-experiment.json` в корне указывает на этот же файл.
- [E01 README](../../experiments/E01-solver-comparison/README.md) — протокол и карта результатов.
- [mirror/ops/idle-audits-v1/campaign_report.md](../../experiments/E01-solver-comparison/runs/deflow-campaign-20261111/revision3/mirror/ops/idle-audits-v1/campaign_report.md) — последняя скопированная действующая сводка. При недоступном сервере это snapshot, а не доказательство текущего состояния.
- [idle-audits-v1/manifest.json](../../experiments/E01-solver-comparison/runs/deflow-campaign-20261111/revision3/ops/idle-audits-v1/manifest.json) — утверждённое 12 ноября выполнение девяти исходных аудитов последовательно на свободной GPU 1, параллельно обучению на других картах. Включает хэши реализации, исходного ответа, входных капсул и сохранённые идентичности родителей.
- [packing-v1/manifest.json](../../experiments/E01-solver-comparison/runs/deflow-campaign-20261111/revision3/ops/packing-v1/manifest.json) — утверждённое 11 ноября размещение максимум двух compute процессов на GPU, точные очереди, идентичности исходных Newton, hashes и критерий canary. Scientific contract и его anchor не изменены.
- [sync_status.json](../../experiments/E01-solver-comparison/runs/deflow-campaign-20261111/revision3/ops/sync_status.json) — локальный результат копирования.

Сервер: `workspace.featurize.cn:18290`, user `featurize`. SSH wrapper и interpreter берутся из профиля и контракта. Credentials остаются в существующем macOS Keychain, в документах их нет.

## Службы

Текущие определения находятся в [launchd](launchd/), логи — рядом с этим README.

| Служба | Label |
| --- | --- |
| Compact sync каждые 120 секунд | `com.codex.deflow-20261111-v3-sync` |
| Task keepawake | `com.codex.deflow-rebuttal-keepawake-v2` |

Sync label сохранён, поскольку frozen code завершает именно его после final acknowledgment.
Теперь label запускает `ops/idle-audits-v1/sync_packed.py`. Он использует исходный
frozen transfer verifier, но завершает sync только после проверенного final
acknowledgment новой очереди. Терминальные файлы исходного координатора не
останавливают этот sync. Исходный координатор не перезапускается и остаётся
родителем текущих Newton до наблюдения их настоящих exit codes. Его
`ops/ADMISSION_BLOCKED.json` — ожидаемый административный запрет будущих
запусков старой очереди. Аналогичный административный запрет установлен для
будущих запусков `packing-v1`, её текущие дети и настоящие exit observers
сохранены. Актуальные ошибки находятся в `ops/idle-audits-v1/`, а ранее
подтверждённая научная ошибка N2 отражается в каноническом статусе.

Raw `audits/*/raw.pt`, все выбранные капсулы и итоговые таблицы добавлены к
проверяемой compact-копии. Научные попытки, исходные failed markers и checkpoints
сохраняются. Перезапусков N2 и других научных повторов нет.

Для проверки:

```bash
launchctl list com.codex.deflow-20261111-v3-sync
launchctl list com.codex.deflow-rebuttal-keepawake-v2
```

После reboot загружать только отсутствующую локальную службу, предварительно проверив terminal markers. Если финальный sync уже принят, заново не загружать. Не запускать повторно server coordinator или обучение.

```bash
launchctl bootstrap "gui/$(id -u)" "$PWD/rebuttal/common/operations/launchd/com.codex.deflow-20261111-v3-sync.plist"
launchctl bootstrap "gui/$(id -u)" "$PWD/rebuttal/common/operations/launchd/com.codex.deflow-rebuttal-keepawake-v2.plist"
```

## Сохранность

Текущая научная кампания сохранена целиком. Frozen contract, источники и серверные пути не менялись. Внешние aliases в старом workspace перенаправлены на новые канонические каталоги, поэтому исторические absolute paths в контракте продолжают работать. Не удалять их во время кампании.

Полный raw revision3 остаётся на сервере. Compact mirror ограничен 22 GiB, local reserve 8 GiB и remote reserve 30 GiB задаются frozen contract. Для локальных копий Mac должен оставаться включённым и иметь сеть. Нет автоматических научных retries/resume.

Журнал упрощения и проверка SHA256 находятся в [archive/organization-20261111](../../archive/organization-20261111/). Старые operations и migration scripts там исторические, их повторно не запускать.

## Дополнительная кампания E02

[Sequential Anderson ×3](../../experiments/E02-sequential-anderson/README.md) использует другой сервер `workspace.featurize.cn:13534`, отдельный frozen manifest и sync. Основной `status.py` выше показывает E01. Для E02: `python3 rebuttal/experiments/E02-sequential-anderson/runs/sa-20261112/ops/status_verified.py`. Не заменять `active_campaign.json` и не останавливать E01 при работе с E02.


## Полный перенос отменён, 12 ноября 2026

По запросу пользователя полный raw-перенос отменён. Новые downloaders и упаковка остановлены, источники сохранены на серверах. Полный перенос не возобновлять без нового запроса. Обычная compact sync продолжает работать.

[ANALYSIS_HANDOFF.md](ANALYSIS_HANDOFF.md) содержит все серверные пути, доступ, run IDs, telemetry/checkpoints, evaluations, audits, masks и научные ограничения. Локальные фрагменты полной копии не верифицированы и не являются источником полной кампании.
