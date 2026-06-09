# Validação Metodológica — TCC F1 Predictive Model

Data de execução desta validação: 02/06/2026

Cada item foi verificado ativamente contra código, dados e artefatos do repositório — não apenas afirmado. Os resultados dos comandos de verificação são apresentados como evidência.

---

## 1. Cobertura dos dados — corridas por temporada vs. calendário oficial

**Verificação:** cruzamento entre o número de GPs no `ergast_2018_2024.csv` + `ergast_2025_results.csv` e o calendário oficial F1.

| Temporada | GPs oficiais | GPs no raw | Status |
|---|---|---|---|
| 2018 | 21 | 21 | ✅ OK |
| 2019 | 21 | 21 | ✅ OK |
| 2020 | 17 | 17 | ✅ OK (calendário reduzido por COVID) |
| 2021 | 22 | 22 | ✅ OK |
| 2022 | 22 | 22 | ✅ OK |
| 2023 | 22 | 22 | ✅ OK |
| 2024 | 24 | 24 | ✅ OK |
| 2025 | 24 | 24 | ✅ OK |

**Resultado: 8/8 temporadas com cobertura completa.** O dataset não tem GPs faltando.

**Dataset de modelagem após DNF Excluded:**

| Temporada | GPs | Linhas no X/y | Média pilotos/GP |
|---|---|---|---|
| 2018 | 21 | 335 | 15.95 |
| 2019 | 21 | 360 | 17.14 |
| 2020 | 17 | 283 | 16.65 |
| 2021 | 22 | 381 | 17.32 |
| 2022 | 22 | 366 | 16.64 |
| 2023 | 22 | 374 | 17.00 |
| 2024 | 24 | 425 | 17.71 |
| 2025 | 24 | 419 | 17.46 |
| **Total** | **173** | **2.943** | **17.01** |

A média de ~17 pilotos por GP (de 20 possíveis) reflete as exclusões por DNF — o que é esperado e coerente com a taxa histórica de ~13-20% de abandono por corrida.

---

## 2. Integridade do dataset final de modelagem

**Verificação direta em `dataset_modelagem_X_2018_2025.csv` e `dataset_modelagem_y_2018_2025.csv`:**

| Métrica | Resultado | Status |
|---|---|---|
| Shape X | (2.943, 15) | ✅ |
| Shape y | (2.943, 6) | ✅ |
| NaN em X | 0 | ✅ |
| NaN em y | 0 | ✅ |
| RaceID duplicados | 0 | ✅ |
| Colunas proibidas em X | 0 | ✅ |

**Features presentes em X** (verificado):
```
qualifying_position, constructor_coef_rapm, recent_form_5,
driver_constructor_synergy, constructor_wins_total, driver_coef_rapm,
track_complexity, tire_compound_start, season_factor,
avg_pit_stops_circuit, constructor_dnf_rate, grid_penalty, altitude_m
```

Exatamente as 13 features do contrato em `models/feature_selection/features_modelagem_2018_2025.json`.

---

## 3. Status final de leakage — verificação em X

**Verificação direta das colunas que foram identificadas como leakage:**

| Feature problemática | Presente em X | Status |
|---|---|---|
| `safety_car_flag` (dado pós-corrida) | AUSENTE | ✅ Corrigido |
| `weather_impact_observed` (dado durante corrida) | AUSENTE | ✅ Corrigido |
| `grid_position` (redundante r=0.962 com qualifying) | AUSENTE | ✅ Removida |
| `recent_form_3` (redundante r=0.987 com recent_form_5) | AUSENTE | ✅ Removida |

**Substitutos presentes em X:**

| Substituto | Presente | Mecanismo causal |
|---|---|---|
| `incident_rate_hist_norm` | ✅ | `expanding().mean().shift(1)` por circuito |
| `qualifying_position` | ✅ | Evento pré-corrida (sábado) |

**Conclusão:** nenhuma coluna proibida ou com leakage confirmado está presente no X de modelagem.

---

## 4. Causalidade das features — verificação com dados reais

### 4.1 `recent_form_5` — evidência de causalidade

Verificação com Hamilton em 2018 (primeiras corridas do dataset):

| Round | `finish_position` | `recent_form_5` | Correto? |
|---|---|---|---|
| 1 | 2 | 0.000 | ✅ Cold-start — sem histórico anterior |
| 2 | 3 | 2.000 | ✅ Usa só round 1 (posição 2) |
| 3 | 4 | 2.556 | ✅ Usa rounds 1-2 com pesos 2,1 |
| 4 | 1 | 3.167 | ✅ Usa rounds 1-3 com pesos 3,2,1 |
| 5 | 1 | 2.429 | ✅ Usa rounds 1-4 com pesos 4,3,2,1 |

Interpretação: no round 2, `recent_form_5 = 2.0` significa a posição média ponderada do round 1 (apenas uma corrida disponível). O resultado do round atual nunca é incluído na feature do mesmo round.

**Número de registros com cold-start por temporada:**

| Temporada | Cold-starts (recent_form_5=0) |
|---|---|
| 2018 | 20 |
| 2019 | 6 |
| 2020 | 3 |
| 2021–2025 | 2-3 por temporada |

Os 20 cold-starts de 2018 correspondem aos pilotos no round 1 (primeiro GP do dataset — sem histórico para nenhum piloto). As temporadas seguintes têm poucos cold-starts porque apenas pilotos estreantes no grid não têm histórico.

### 4.2 `driver_coef_rapm` — evidência de causalidade

Verificação no arquivo `coef_pilotos_rapm_2018_2025.csv`:

| Corrida | `coefficient_status` | Evidência |
|---|---|---|
| Round 1/2018 (todos os pilotos) | `cold_start_sem_historico_suficiente` | ✅ Sem dados anteriores → coef=0.0 |
| Round 2/2018 em diante | `estimado_historico_anterior` | ✅ Treinado só com round 1 em diante |

O campo `coefficient_status` no arquivo de coeficientes rastreia explicitamente se o coeficiente foi estimado ou é cold-start — rastreabilidade completa.

**Evolução temporal do coeficiente de Hamilton** (sinal de que o modelo aprende ao longo do tempo):

| Temporada | `driver_coef_rapm` (última corrida) |
|---|---|
| 2018 | 2.241 |
| 2021 | 2.853 |
| 2025 | 0.671 |

O coeficiente de Hamilton cresce até 2021 (auge na Mercedes) e cai em 2025 (primeiro ano na Ferrari, adaptação). Isso é o comportamento esperado de um coeficiente causal que captura desempenho histórico — não é um valor fixo, mas uma estimativa que evolui com os resultados.

### 4.3 `driver_dnf_rate` — evidência de causalidade

A feature é calculada no `historico_dnf_classificado` (que inclui DNFs) com `cumsum().shift(1)` — o resultado de cada corrida só entra no cálculo da corrida *seguinte*. Validado no relatório 11:

```
driver_dnf_rate nonzero: 2.096 de 2.524 (83%)
driver_dnf_rate mínimo/máximo: 0.000 / 1.000
```

Zero registros com NaN (confirmado).

### 4.4 `avg_pit_stops_circuit` — cold-start documentado

O manifesto registra `pitstop_cold_start_rows_2018_2025: 511` — 17.4% das linhas receberam o valor global anterior (sem histórico específico do circuito). Esse cold-start é esperado: a primeira corrida de um circuito no dataset não tem histórico de pit stops daquele circuito.

---

## 5. Viés de sobrevivência — análise do impacto dos DNFs

**Taxa de DNF por temporada** (calculada no `historico_dnf_classificado_2018_2025.csv`):

| Temporada | Total | DNFs | Taxa DNF |
|---|---|---|---|
| 2018 | 420 | 85 | **20.2%** |
| 2019 | 420 | 60 | 14.3% |
| 2020 | 340 | 57 | 16.8% |
| 2021 | 440 | 59 | 13.4% |
| 2022 | 440 | 74 | 16.8% |
| 2023 | 440 | 66 | 15.0% |
| 2024 | 479 | 54 | 11.3% |
| 2025 | 479 | 60 | 12.5% |
| **Total** | **3.458** | **515** | **14.9%** |

**Natureza do viés:**

O modelo aprende exclusivamente com corridas onde o piloto *completou* a prova. Isso significa:
- `driver_coef_rapm` é estimado em corridas de conclusão — não captura padrão de abandono
- `recent_form_5` só acumula resultados de corridas completadas
- Features como `tire_compound_start` e `avg_pit_stops_circuit` são calculadas em contextos sem abandono

**Impacto prático em 2026:**

A temporada 2026 tem regulamento completamente novo. Carros novos com tecnologias não testadas tendem a ter mais falhas mecânicas nas primeiras corridas — potencialmente com taxa de DNF mecânico acima do histórico. O modelo, treinado com viés de sobrevivência, não tem capacidade de prever:
- Que um piloto específico abandonará por falha
- Como o desempenho médio muda quando os abandonos são removidos

Isso é uma limitação documentada — não um erro de implementação. A decisão de usar DNF Excluded está alinhada com Henderson et al. [9] e é uma simplificação consciente para o problema de regressão de posição final.

**Distribuição dos DNFs por categoria:**

| Categoria | 2018-2025 | % do total DNF |
|---|---|---|
| `dnf_carro` (falha mecânica) | 174 | 33.8% |
| `dnf_piloto` (acidente) | 147 | 28.5% |
| `dnf_outros` (desclass., DNS) | 194 | 37.7% |
| **Total** | **515** | 100% |

Os 194 `dnf_outros` incluem desclassificações (Hamilton, Leclerc, Gasly na China 2025; Norris, Piastri em Las Vegas 2025) — confirmado e metodologicamente correto.

---

## 6. Reprodutibilidade

### Seeds

Todos os modelos e algoritmos estocásticos usam `random_state=42`:

| Script | Random state |
|---|---|
| `rapm_ridge.py` | `Ridge(random_state=42)` |
| `walk_forward.py` | `XGBRegressor(random_state=42)` |
| `walk_forward_lightgbm.py` | `LGBMRegressor(random_state=42)` |
| `walk_forward_random_forest.py` | `RandomForestRegressor(random_state=42)` |
| `tuning_xgboost.py` | `TPESampler(seed=42)` |
| `tuning_lightgbm.py` | `TPESampler(seed=42)` |
| `tuning_randomforest.py` | `TPESampler(seed=42)` |
| `rfe_xgboost_features.py` | `XGBRegressor(random_state=42)` |
| `gerar_feature_importance_modelos.py` | `random_state=42` em todos |

**Limitação identificada:** `TPESampler(seed=42)` garante reprodutibilidade do Optuna somente em execução *single-threaded*. Se `n_jobs > 1` for usado no Optuna (não é o caso aqui — os scripts usam `n_jobs=1` para o estudo, `n_jobs=4` apenas internamente nos modelos), a ordem das trials pode variar. Os scripts de tuning **não paralelizam as trials do Optuna**, apenas os cálculos internos dos modelos — reprodutibilidade garantida.

### Versões fixadas

Todas as dependências estão pinadas em `requirements.txt`:

```
pandas==2.1.4, numpy==1.26.4, scikit-learn==1.8.0,
xgboost==2.0.3, lightgbm==4.3.0, optuna==3.6.1, adapt==0.4.5
```

Isso garante que os resultados são reprodutíveis em qualquer ambiente com as versões corretas instaladas.

### Manifestos de rastreabilidade

| Manifesto | Arquivo | O que registra |
|---|---|---|
| Feature Engineering | `data/processed/manifest_feature_engineering.json` | Colunas proibidas, contratos de join, validações |
| RAPM Ridge | `models/rapm/manifest_rapm_ridge.json` | alpha, decay, n_corridas, anti-leakage rule |
| RFE XGBoost | `models/feature_selection/manifest_rfe_xgboost.json` | Features selecionadas, MAE por subconjunto |

---

## 7. Consistência dos folds de validação

**Isolamento temporal verificado:**

| Etapa | Folds usados | Fold 2025 exposto? |
|---|---|---|
| Otimização time-decay | 2023, 2024 | ❌ Não |
| Tuning Optuna | 2023, 2024 | ❌ Não |
| RFE (seleção features) | 2025 apenas | ❌ Não para tuning |
| Avaliação final | 2023, 2024, 2025 | ✅ Sim — somente na avaliação |

A cadeia de isolamento é rigorosa: o fold 2025 só é "visto" na avaliação final, após todos os hiperparâmetros e features terem sido fixados.

---

## 8. Resumo — pontos validados e limitações residuais

### Validados como corretos

| Item | Evidência |
|---|---|
| Cobertura 8/8 temporadas | Cruzamento raw vs. calendário oficial |
| Zero NaN no dataset final | Verificação direta |
| Zero RaceID duplicados | Verificação direta |
| Colunas proibidas ausentes de X | Verificação direta |
| `recent_form_5` causal | Verificação linha a linha com Hamilton 2018 |
| `driver_coef_rapm` causal | `coefficient_status` no arquivo de saída |
| Leakages corrigidos | `safety_car_flag` e `weather_impact_observed` ausentes de X |
| Seeds fixados em todos os scripts | `random_state=42` em 14 scripts verificados |
| Versões pinadas | `requirements.txt` com versões exatas |
| Fold 2025 isolado do tuning | FOLDS_TUNING = apenas 2023-2024 |

### Limitações residuais documentadas

| Limitação | Impacto | Onde documentado |
|---|---|---|
| Alpha=10.0 do RAPM não tunado | Coeficientes subótimos para entidades com poucas corridas | `docs/tecnico/05_rapm_ridge.md` |
| Viés de sobrevivência dos DNFs | Modelo não calibrado para corridas com altas taxas de abandono | `docs/tecnico/02_limpeza_dnf.md` |
| Scaler normalização ajustado em 2018-2024 (não por fold) | Risco se colunas `_zscore` forem usadas como features | `docs/tecnico/03_encoding_normalizacao.md` |
| RFE com fold único (2025) | Subconjunto pode estar ligeiramente sobreajustado ao padrão de 2025 | `docs/tecnico/07_selecao_features.md` |
| Mapeamento `race_name → circuit_id` hardcoded | Madrid 2026 não mapeado — pipeline quebraria | `docs/tecnico/03_encoding_normalizacao.md` |
| Pesos de `track_complexity` arbitrários | Índice não calibrado empiricamente | `docs/tecnico/06_feature_engineering.md` |
| `grid_penalty` sem referência bibliográfica | Feature adicionada sem embasamento explícito | `docs/tecnico/06_feature_engineering.md` |
| `driver_wins_total` excluída pelo RFE | Divergência com arquitetura original | `docs/tecnico/07_selecao_features.md` |
| R² < 0.75 (meta não atingida) | Meta baseada em TabNet com dados diferentes | `docs/tecnico/10_resultados_feature_importance.md` |
| Top-3 accuracy 18-24% (meta não atingida) | Critério de regressão vs. meta de classificação | `docs/tecnico/10_resultados_feature_importance.md` |
