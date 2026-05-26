# Lista canônica de features do modelo — F1 Predictive Model

Data: 25/05/2026
Fonte de referência: `ArquiteturaProposta.pdf`, `manifest_feature_engineering.json`
Base de entrada: `data/processed/dataset_feature_engineering_ready_2018_2025.csv`

Atualização metodológica: após a análise de correlação, leakage e RFE temporal com
XGBoost em 25/05/2026, o conjunto de entrada do modelo foi reduzido de 21 para
15 features. `safety_car_flag` permanece apenas como auditoria e foi substituída
por `incident_rate_hist_norm`; `weather_impact_factor` passou a ser histórico
causal por circuito, mas ficou fora de X após a RFE; `recent_form_3` e
`grid_position` foram removidas por redundância severa.

---

## Target

| Coluna | Tipo | Fonte |
|---|---|---|
| `finish_position` | Numérica (1–20) | Ergast/Jolpica `results.positionOrder` |

`finish_position` permanece na base FE-ready como target e como insumo histórico
para features causais. Deve ser separado de `X` antes de qualquer treino.

---

## Colunas proibidas em X (data leakage)

| Coluna | Motivo |
|---|---|
| `finish_position` | É o target — nunca entra em X |
| `points` | Calculado do resultado — leakage direto |
| `race_points` | Idem |
| `fastest_lap_race` | Dado pós-corrida — leakage direto |
| `previous_position` | Captura padrão trivial que quebra em 2026 |

---

## Features que entram em X — lista final

### Categoria: Grid

| Feature | Tipo | Descrição | Fonte | Status |
|---|---|---|---|---|
| `qualifying_position` | Numérica | Posição após qualifying; fallback = grid_position | FastF1 qualifying | Pronto |
| `grid_penalty` | Numérica | grid_position − qualifying_position | Derivado etapa 07 | Pronto |

### Categoria: Forma recente (a calcular na FE)

| Feature | Tipo | Descrição | Fonte | Status |
|---|---|---|---|---|
| `recent_form_5` | Numérica | Média ponderada últimas 5 corridas (pesos 5,4,3,2,1) | finish_position histórico + shift(1) | **Pendente FE** |

### Categoria: Piloto (a calcular na FE)

| Feature | Tipo | Descrição | Fonte | Status |
|---|---|---|---|---|
| `driver_coef_rapm` | Numérica | Coeficiente Ridge time-decay do piloto | `rapm_ridge.py` — quinta 20/05 | **Pendente RAPM** |
| `driver_dnf_rate` | Numérica | DNFs piloto / corridas disputadas (causal) | `historico_dnf_classificado_2018_2025.csv` + shift(1) | **Pendente FE** |

### Categoria: Construtor (a calcular na FE)

| Feature | Tipo | Descrição | Fonte | Status |
|---|---|---|---|---|
| `constructor_coef_rapm` | Numérica | Coeficiente Ridge time-decay do construtor | `rapm_ridge.py` — quinta 20/05 | **Pendente RAPM** |
| `constructor_dnf_rate` | Numérica | DNFs mecânicos / corridas do construtor (causal) | `historico_dnf_classificado_2018_2025.csv` + shift(1) | **Pendente FE** |
| `constructor_wins_total` | Numérica | Total de vitórias do construtor até aquela data | Ergast acumulado + shift(1) | **Pendente FE** |

### Categoria: Sinergia (a calcular na FE)

| Feature | Tipo | Descrição | Fonte | Status |
|---|---|---|---|---|
| `driver_constructor_synergy` | Numérica | Média histórica do piloto com essa equipe específica (causal) | Ergast histórico + shift(1) | **Pendente FE** |

### Categoria: Circuito

| Feature | Tipo | Descrição | Fonte | Status |
|---|---|---|---|---|
| `track_complexity` | Numérica [0,1] | Score combinado: curvas, comprimento, altitude, tipo, incidentes históricos | `circuitos_manual.csv` + etapa 09 | Pronto |
| `altitude_m` | Numérica | Altitude do circuito em metros | `circuitos_manual.csv` | Pronto |

### Categoria: Pneu

| Feature | Tipo | Descrição | Fonte | Status |
|---|---|---|---|---|
| `tire_compound_start` | Ordinal (0–6) | Composto de largada (ordinal: HARD=1 … HYPERSOFT=6) | FastF1 `Compound` | Pronto |
| `avg_pit_stops_circuit` | Numérica | Média histórica causal de pit stops naquele circuito | Ergast pitstops + expanding causal | Pronto |

### Categoria: Temporada

| Feature | Tipo | Descrição | Fonte | Status |
|---|---|---|---|---|
| `season_factor` | Inteira | Ano da corrida — captura evolução tecnológica | Derivado de `season` | Pronto |

### Categoria: Eventos de pista

| Feature | Tipo | Descrição | Fonte | Status |
|---|---|---|---|---|
| `incident_rate_hist_norm` | Numérica [0,1] | Taxa histórica normalizada de SC/VSC no circuito antes da corrida | FastF1 TrackStatus 4/6/7 + shift(1) | Pronto |

---

## Features prontas na base FE-ready

Das 15 features finais, **8 já estão prontas** em
`data/processed/dataset_feature_engineering_ready_2018_2025.csv`:

- `qualifying_position`, `grid_penalty`
- `track_complexity`, `altitude_m`
- `tire_compound_start`, `avg_pit_stops_circuit`
- `season_factor`
- `incident_rate_hist_norm`

**7 features ainda precisam ser calculadas** em `feature_engineering.py`:

- `recent_form_5` (forma recente)
- `driver_coef_rapm`, `constructor_coef_rapm` (via `rapm_ridge.py`, quinta 20/05)
- `driver_dnf_rate`, `constructor_dnf_rate` (via `historico_dnf_classificado_2018_2025.csv`)
- `constructor_wins_total` (acumulado causal)
- `driver_constructor_synergy` (média histórica por par piloto-equipe)

---

## Colunas auxiliares na base FE-ready (não entram em X)

| Coluna | Uso |
|---|---|
| `finish_position` | Target e insumo histórico causal |
| `driver_id`, `constructor_id`, `circuit_id` | Chaves de join para feature engineering |
| `season`, `round`, `RaceID`, `race_name` | Ordenação temporal e rastreabilidade |
| `laps`, `status` | Contexto de corrida |
| `compound_ordinal` | Origem de `tire_compound_start` |
| `fastf1_avg_lap_time`, `fastf1_avg_sector*` | Insumos de auditoria (não features do modelo) |
| `track_complexity_static` | Versão estática para comparação de importância |
| `incident_rate_hist` | Componente bruto de auditoria para taxa de SC/VSC |
| `safety_car_flag` | Evento real da corrida; auditoria/outlier, não entra em X |
| `weather_impact_observed` | Clima real observado da corrida; auditoria, não entra em X |
| `weather_impact_cold_start_flag` | Flag de cold-start da feature causal de clima |
| `outlier_*_flag` | Flags de rastreabilidade metodológica |
| `avg_pit_stops_circuit_static_global` | Auditoria (não usar como feature) |
| Colunas One-Hot (`circuito_*`, `constructor_*`) | Encoding intermediário (não entram no modelo tree-based) |
| Colunas Z-score e MinMax (`*_zscore`, `*_minmax`) | Artefatos de pré-processamento para Ridge Regression baseline |

---

## Regras de construção de X

```python
COLUNAS_PROIBIDAS = [
    "finish_position", "points", "race_points",
    "fastest_lap_race", "previous_position",
]

FEATURES_FINAIS = [
    # Grid
    "qualifying_position", "grid_penalty",
    # Forma recente (calcular na FE)
    "recent_form_5",
    # Piloto (calcular na FE)
    "driver_coef_rapm", "driver_dnf_rate",
    # Construtor (calcular na FE)
    "constructor_coef_rapm", "constructor_dnf_rate", "constructor_wins_total",
    # Sinergia (calcular na FE)
    "driver_constructor_synergy",
    # Circuito
    "track_complexity", "altitude_m",
    # Pneu
    "tire_compound_start", "avg_pit_stops_circuit",
    # Temporada
    "season_factor",
    # Eventos
    "incident_rate_hist_norm",
]

# Para o baseline Ridge Regression, adicionar as colunas normalizadas:
FEATURES_RIDGE_EXTRA = [
    "grid_position_minmax",
    "fastf1_avg_lap_time_zscore",
    "fastf1_best_lap_time_zscore",
]
```

---

## Correlações esperadas com o target (referência da arquitetura)

| Feature | Correlação esperada (r) | Referência |
|---|---|---|
| `qualifying_position` | ≈ 0,71 | Advanced ML paper (Barra et al.) |
| `constructor_coef_rapm` | ≈ 0,60 | RAPM paper — carro explica 64% da variância |
| `recent_form_5` | ≈ 0,55–0,65 | Estimativa baseada na literatura |
| `driver_coef_rapm` | ≈ 0,35 | RAPM paper — piloto explica 36% |
| `driver_dnf_rate` | ≈ 0,25–0,35 | RF+SHAP paper (Ruan et al.) |

---

## Análise de correlação entre features

Pares avaliados ou esperados com alta correlação (r > 0,85 → considerar remover uma):

| Par | Correlação esperada | Ação se r > 0,85 |
|---|---|---|
| `driver_wins_total` vs `driver_experience` | Alta | Manter `driver_wins_total`, remover `driver_experience` |
| `constructor_coef_rapm` vs `constructor_wins_total` | Alta | Manter `constructor_coef_rapm`, remover `constructor_wins_total` |
| `track_complexity` vs `altitude_m` | Moderada | Manter ambas (capturam aspectos distintos) |
| `recent_form_3` vs `recent_form_5` | Alta observada (`r=0,9874`) | Remover `recent_form_3`; manter `recent_form_5` |
| `grid_position` vs `qualifying_position` | Alta observada (`r=0,9616`) | Remover `grid_position`; manter `qualifying_position` + `grid_penalty` |

## Decisão RFE XGBoost

RFE temporal executada com treino em 2018-2024 e validação em 2025:

- Melhor subconjunto: 15 features
- MAE 2025: 2,466866
- Features removidas pela RFE: `driver_wins_total`, `driver_experience`,
  `weather_impact_factor`, `circuit_type`

Artefatos:

- `models/feature_selection/rfe_xgboost_ranking.csv`
- `models/feature_selection/rfe_xgboost_subsets.csv`
- `models/feature_selection/relatorio_rfe_xgboost.txt`
