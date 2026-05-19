# Inconsistências entre `dados_necessarios.txt` e `data/raw/`

Data da verificação: 2026-05-19

---

## Resumo executivo

Dos 13 grupos de dados listados em `dados_necessarios.txt`, apenas **4** foram parcialmente extraídos. Os outros 9 estão ausentes por completo ou com cobertura crítica faltando. Há ainda campos específicos ausentes dentro dos arquivos já extraídos.

**Nota:** O `dados_necessarios.txt` lista OpenF1 com escopo 2023–2026, mas o escopo real do projeto é **2025–2026** para OpenF1 (FastF1 cobre 2018–2024). Esta inconsistência entre o documento e a arquitetura real está refletida nas seções abaixo.

---

## 1. Arquivos extraídos e seus problemas

### 1.1 `ergast_2018_2024.csv` + `ergast_2025_results.csv` — Jolpica `/results/`

**Correspondência:** seção 1 do `dados_necessarios.txt`

| Problema | Detalhe |
|---|---|
| Campo `constructor` usa nome, não ID | O script salva `Constructor["name"]` (ex: "Ferrari") em vez de `Constructor["constructorId"]` (ex: "ferrari"). O `dados_necessarios` exige `constructorId` como chave de join. |
| Campo `laps` ausente | `Results[].laps` (voltas completadas) não foi extraído. O script ignora esse campo completamente. |
| ~~Campo `finish_position` usa `position`, não `positionOrder`~~ | Falso positivo: a API Jolpica não expõe o campo `positionOrder` no JSON. O campo `position` já retorna a ordem numérica correta inclusive para DNFs (`"position": "16"`, `"positionText": "R"`). Sem inconsistência aqui. |
| Cobertura 2025 incompleta | `ergast_2025_results.csv` existe, mas não cobre corridas após a data de extração (temporada em andamento — verificar até qual rodada foi extraído). |

---

### 1.2 `ergast_pitstop_2018_2024.csv` — Jolpica `/pitstops/`

**Correspondência:** seção 3 do `dados_necessarios.txt`

| Problema | Detalhe |
|---|---|
| Pitstops de 2025 ausentes | O arquivo só cobre 2018–2024. Nenhum script extrai pitstops de 2025. |
| Campo `time` (horário do pit) ausente | O script extrai `stop`, `lap`, `duration` mas não o campo `time` do JSON. Impacto baixo pois o `lap` já identifica o contexto. |

---

### 1.3 `fastf1_laps_2018_2024.csv` — FastF1 `session.laps`

**Correspondência:** seção 6 do `dados_necessarios.txt`

Colunas presentes: `Driver, LapNumber, LapTime, Sector1Time, Sector2Time, Sector3Time, Compound, TyreLife, Stint, season, round`

| Problema | Detalhe |
|---|---|
| `TrackStatus` ausente | Essencial para derivar `safety_car_flag` (valor "4"=SC, "5"=VSC). Não foi incluído no script de extração. |
| `FreshTyre` ausente | Necessário para confirmar se o pneu era novo na largada (stint 1, volta 1). |
| `PitInTime` / `PitOutTime` ausentes | Necessários para confirmar número de pit stops via FastF1. |
| FastF1 2025 ausente | O `fastf1_checkpoint.json` não tem nenhuma entrada de 2025. O `dados_necessarios` pede 2018–2025. |

---

### 1.4 `fastf1_qualifying_2018_2024.csv` — FastF1 qualifying

**Correspondência:** seção 2 do `dados_necessarios.txt` (FastF1 é a fonte correta para qualifying)

Colunas presentes: `Driver, LapTime, Sector1Time, Sector2Time, Sector3Time, season, round`

| Problema | Detalhe |
|---|---|
| Colunas Q1, Q2, Q3 ausentes | O CSV tem `LapTime` único (melhor volta), não os tempos separados por sessão Q1/Q2/Q3. Impede a imputação KNN de nulos de qualifying conforme especificado. |
| Coluna `position` ausente | A posição de qualifying (para cruzar com `grid_position`) não está presente. |
| Qualifying 2025 ausente | Nenhum dado de qualifying para 2025. |

---

### 1.5 `openf1_starting_grid_2025.csv` — OpenF1 `/starting_grid`

**Correspondência:** parcialmente seção 12 (session_result) do `dados_necessarios.txt`

| Problema | Detalhe |
|---|---|
| Não é `/session_result` | O arquivo extrai grid de largada, não o resultado final da corrida (`finish_position`, `dnf`, `number_of_laps`). |
| `driver_number` como chave, não `driverId` | OpenF1 usa número do carro; Jolpica usa `driverId`. O join requer tabela auxiliar de mapeamento (`mapear_openf1_ergast.py` existe, mas não está integrado ao pipeline). |

---

## 2. Dados completamente ausentes (nenhum arquivo extraído)

### 2.1 Jolpica `/circuits/` — seção 4

Nenhum CSV de circuitos foi extraído. Campos ausentes:
- `circuitId` (chave de circuito para joins)
- `circuitName`
- `Location.lat` / `Location.long`
- `Location.country`

**Impacto:** Sem chave de circuito, qualquer join por circuito depende de `race_name` (texto livre, frágil — ex: "70th Anniversary Grand Prix" e "British Grand Prix" são corridas diferentes no mesmo circuito).

---

### 2.2 Jolpica `/drivers/` — seção 5

Nenhum CSV de pilotos foi extraído. Campo ausente:
- `dateOfBirth` (metadado de piloto para joins futuros)

**Impacto:** Baixo impacto direto — `driver_experience` pode ser derivado por contagem de aparições nos resultados. Mas a ausência de metadados completos de piloto pode dificultar joins futuros.

---

### 2.3 FastF1 `session.laps.get_weather_data()` — seção 7

Nenhum CSV de dados meteorológicos via FastF1 foi criado. Campos ausentes:
- `AirTemp`, `Humidity`, `Rainfall`, `TrackTemp`, `WindSpeed`

**Impacto:** A feature `weather_impact_factor` não tem dados para 2018–2024. É a lacuna de maior prioridade.

---

### 2.4 OpenF1 `/stints` (2025–2026) — seção 8

Nenhum CSV de stints via OpenF1. Campos ausentes:
- `compound` (stint_number==1 → `tire_compound_start` para 2025+)
- `tyre_age_at_start`
- `stint_number`

**Impacto:** `tire_compound_start` para 2025 não tem fonte (FastF1 cobre até 2024).

---

### 2.5 OpenF1 `/weather` (2025–2026) — seção 9

Nenhum CSV de clima via OpenF1. Campos ausentes:
- `air_temperature`, `humidity`, `rainfall`, `track_temperature`, `wind_speed`

**Impacto:** `weather_impact_factor` para 2025 sem fonte (FastF1 cobre até 2024).

---

### 2.6 OpenF1 `/race_control` (2025–2026) — seção 10

Nenhum CSV de race control. Campos ausentes:
- `category`, `message`, `flag`

**Impacto:** `safety_car_flag` para 2025 sem fonte. Para 2018–2024, o `TrackStatus` do FastF1 ainda está ausente (ver 1.3) — ou seja, a feature não tem dados em nenhum período.

---

### 2.7 OpenF1 `/meetings` (2025–2026) — seção 11

Nenhum CSV de meetings. Campos ausentes:
- `circuit_type` ("Permanent" / "Temporary - Street") para 2025+
- `meeting_key`, `circuit_short_name`

**Impacto:** `circuit_type` para 2025 não tem fonte. Para 2018–2024, a tabela manual (seção 13) é a fonte — mas ela também não foi criada.

---

### 2.8 OpenF1 `/session_result` (2025–2026) — seção 12

Nenhum CSV de session_result. Campos ausentes:
- `position` (finish_position alternativa para 2025+)
- `dnf` (flag booleana)
- `number_of_laps`

**Impacto:** Flag de DNF não existe em nenhum arquivo extraído. Para 2018–2024, o campo `status` do Jolpica permite derivá-la, mas a flag direta `dnf` do OpenF1 para 2025 não está disponível.

---

### 2.9 Tabela manual — seção 13

Nenhum arquivo de tabela manual foi criado. Campos ausentes:
- `altitude` por circuito (~25 circuitos)
- `number_of_corners` por circuito
- `circuit_length_km` por circuito
- `circuit_type` para 2018–2024 (OpenF1 só cobre 2025+)

**Impacto:** A feature `track_complexity` não tem nenhuma fonte de dados. `altitude` e `circuit_type` histórico estão completamente ausentes.

---

## 3. Problemas de cobertura temporal

| Dado | Escopo previsto | Extraído |
|---|---|---|
| Jolpica results | 2018–2025 | 2018–2024 + 2025 (parcial) |
| Jolpica pitstops | 2018–2025 | 2018–2024 |
| Jolpica circuits | uma vez | **Nenhum** |
| Jolpica drivers | uma vez | **Nenhum** |
| FastF1 laps | 2018–2025 | 2018–2024 (sem 2025) |
| FastF1 qualifying | 2018–2025 | 2018–2024 (sem 2025) |
| FastF1 weather | 2018–2025 | **Nenhum** |
| OpenF1 stints | 2025–2026 | **Nenhum** |
| OpenF1 weather | 2025–2026 | **Nenhum** |
| OpenF1 race_control | 2025–2026 | **Nenhum** |
| OpenF1 meetings | 2025–2026 | **Nenhum** |
| OpenF1 session_result | 2025–2026 | **Nenhum** |
| OpenF1 starting_grid | 2025–2026 | 2025 (sem 2026 — futuro) |
| Tabela manual | todos os circuitos ativos | **Nenhum** |

---

## 4. Problemas de schema / chaves de join

| Problema | Arquivo | Impacto |
|---|---|---|
| `constructor` salvo como nome ("Ferrari"), não como ID ("ferrari") | `ergast_2018_2024.csv`, `ergast_2025_results.csv` | Join por `constructorId` quebra; features de constructor (wins, DNF rate, RAPM) afetadas |
| `driver_number` (OpenF1) vs `driverId` (Jolpica) sem mapeamento integrado | `openf1_starting_grid_2025.csv` | `mapear_openf1_ergast.py` existe mas não está no pipeline |
| Circuito identificado por `race_name` (texto livre) em vez de `circuitId` | Todos os arquivos Jolpica | Joins por circuito são frágeis (ex: dois GPs em Silverstone com nomes diferentes) |
| ~~`finish_position` usa `position` em vez de `positionOrder`~~ | — | Falso positivo: `positionOrder` não existe no JSON da API. `position` já é o campo correto. |

---

## 5. Nota sobre o `openf1_validation.json`

O arquivo é uma verificação de conectividade/amostragem, não uma extração. Confirma que os endpoints OpenF1 funcionam, mas **não gera nenhum dado utilizável para o modelo**. Os dados de stints, weather, race_control, meetings e session_result precisam ser extraídos com scripts dedicados.

---

## 6. Prioridade de extração sugerida

| Prioridade | O que extrair | Features impactadas |
|---|---|---|
| 1 | FastF1 `TrackStatus` (adicionar ao script existente) | `safety_car_flag` 2018–2024 |
| 1 | FastF1 weather (2018–2024) | `weather_impact_factor` 2018–2024 |
| 1 | OpenF1 race_control (2025) | `safety_car_flag` 2025 |
| 1 | OpenF1 weather (2025) | `weather_impact_factor` 2025 |
| 2 | OpenF1 stints (2025) | `tire_compound_start` 2025 |
| 2 | OpenF1 session_result (2025) | `finish_position` + `dnf` 2025 |
| 2 | OpenF1 meetings (2025) | `circuit_type` 2025 |
| 3 | Tabela manual (altitude, corners, length, circuit_type 2018–2024) | `track_complexity`, `altitude`, `circuit_type` |
| 3 | Jolpica circuits | `circuitId` como chave de join |
| 4 | Adicionar Q1/Q2/Q3 e `position` ao script FastF1 qualifying | imputação KNN, `qualifying_position` |
| 4 | Corrigir `constructor` → `constructorId` nos scripts ergast | todas as features de constructor |
| 4 | Adicionar `laps`, `FreshTyre`, `PitInTime`/`PitOutTime` ao script ergast/FastF1 | validação de pit stops, pneu novo |
| 5 | FastF1 2025 + Jolpica pitstops 2025 + Ergast results 2025 completo | cobertura temporal 2025 |
