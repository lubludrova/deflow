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
- [mirror/ops/packing-v1/campaign_report.md](../../experiments/E01-solver-comparison/runs/deflow-campaign-20261111/revision3/mirror/ops/packing-v1/campaign_report.md) — последняя скопированная действующая сводка. При недоступном сервере это snapshot, а не доказательство текущего состояния.
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
Теперь label запускает `ops/packing-v1/sync_packed.py`. Он использует исходный
frozen transfer verifier, но завершает sync только после проверенного final
acknowledgment новой очереди. Терминальные файлы исходного координатора не
останавливают этот sync. Исходный координатор не перезапускается и остаётся
родителем текущих Newton до наблюдения их настоящих exit codes. Его
`ops/ADMISSION_BLOCKED.json` — ожидаемый административный запрет будущих
запусков старой очереди. Актуальные ошибки находятся в `ops/packing-v1/`.

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
