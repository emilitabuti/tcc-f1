# Mapeamento OpenF1 → Ergast/Jolpica

## Objetivo

Este documento descreve como os campos da OpenF1 se relacionam com o schema usado a partir da Jolpica/Ergast.

---

## Arquivos verificados

| Fonte | Arquivo |
|---|---|
| Jolpica/Ergast | `data/raw/ergast_2018_2024.csv` |
| OpenF1 Validation | `data/raw/openf1_validation.json` |
| OpenF1 Starting Grid | `data/raw/openf1_starting_grid_2025.csv` |

---

## Colunas encontradas nos arquivos

### Jolpica/Ergast

```text
['season', 'round', 'race_name', 'driver_id', 'constructor_id', 'grid_position', 'finish_position', 'status', 'points', 'laps']
```

### OpenF1 Position

```text
['session_key', 'driver_number', 'position', 'date', 'meeting_key']
```

### OpenF1 Starting Grid

```text
['season', 'meeting_key', 'race_session_key', 'qualifying_session_key', 'circuit_short_name', 'date_start', 'driver_number', 'grid_position', 'qualifying_lap_duration']
```

---

## Verificação automática dos campos críticos

| Campo crítico | Ergast/Jolpica | OpenF1 Position | OpenF1 Starting Grid |
|---|---|---|---|
| `session_key` | AUSENTE | OK | AUSENTE |
| `driver_number` | AUSENTE | OK | OK |
| `position` | AUSENTE | OK | AUSENTE |
| `date` | AUSENTE | OK | AUSENTE |
| `grid_position` | OK | AUSENTE | OK |

---

## Verificação de campos adicionais

| Campo | Ergast/Jolpica | OpenF1 Position | OpenF1 Starting Grid |
|---|---|---|---|
| `driver_id` | OK | AUSENTE | AUSENTE |
| `round` | OK | AUSENTE | AUSENTE |
| `season` | OK | AUSENTE | OK |
| `finish_position` | OK | AUSENTE | AUSENTE |
| `status` | OK | AUSENTE | AUSENTE |
| `points` | OK | AUSENTE | AUSENTE |
| `meeting_key` | AUSENTE | OK | OK |
| `race_session_key` | AUSENTE | AUSENTE | OK |
| `qualifying_session_key` | AUSENTE | AUSENTE | OK |
| `qualifying_lap_duration` | AUSENTE | AUSENTE | OK |

---

## Tabela de equivalência

| Campo OpenF1 | Equivalente Ergast/Jolpica | Observação |
|---|---|---|
| `driver_number` | `driver_id` | Precisa de lookup número → id do piloto |
| `position` | `finish_position` | Position temporal da OpenF1 |
| `grid_position` | `grid_position` | Starting grid — via Ergast (direto) |
| `position` (starting_grid) | `grid_position` | ⚠️ Na OpenF1 `/starting_grid` o campo chama-se `position`, não `grid_position`. Join por `meeting_key` obrigatório |
| `session_key` | `season + round` | Chave da sessão |
| `meeting_key` | `season + round` | Chave do GP |
| `date` | calendário da corrida | Timestamp temporal |
| `driver_number` | `driver_id` | Lookup necessário: número → driverId (ex: 1 → verstappen). Tabela em `openf1_starting_grid_2025.csv` cruzada com FastF1 |

---

## Estratégia de fallback

| Campo | Fonte principal | Fallback |
|---|---|---|
| `grid_position` | Jolpica/Ergast | OpenF1 `/starting_grid` |
| `finish_position` | Jolpica/Ergast | OpenF1 `/session_result` |
| `status` | Jolpica/Ergast | OpenF1 `/session_result` |
| `points` | Jolpica/Ergast | Sem fallback |

---

## Conclusão

As fontes serão integradas pela seguinte chave composta:

- **Ergast → OpenF1**: `season` + `round` → `meeting_key` (via calendário)
- **Dentro da OpenF1**: `meeting_key` + `session_key` identifica a sessão
- **Piloto**: `driver_id` (Ergast) ↔ `driver_number` (OpenF1) via lookup FastF1

⚠️ **Atenção ao campo de grid**: na OpenF1 `/starting_grid`, a posição de largada está no campo `position` (não `grid_position`). O campo `grid_position` do schema Ergast corresponde ao campo `position` da sessão de qualifying da OpenF1.

⚠️ **Cobertura OpenF1**: endpoints validados para 2023–2025. Para dados históricos (2018–2022) usar exclusivamente Ergast + FastF1.
