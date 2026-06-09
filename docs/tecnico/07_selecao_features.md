# 07 — Seleção de Features

## Contexto

Com todas as features criadas, o conjunto original era de 19 candidatas. O processo de seleção tem dois objetivos: (1) remover features que introduzem leakage ou multicolinearidade severa; (2) encontrar o subconjunto que maximiza a performance preditiva no conjunto de validação temporal.

O processo seguiu três etapas em sequência:

1. **Identificação e correção de leakage** — remoção de dados pós-corrida (25/05/2026).
2. **Análise de correlação** — remoção de redundâncias severas (r > 0.85).
3. **RFE temporal multi-métrica com XGBoost** — seleção do subconjunto ótimo por score composto médio nos folds 2023, 2024 e 2025.

---

## Fundamentação bibliográfica

| Decisão | Referência |
|---|---|
| Critério r > 0.85 para multicolinearidade | Arquitetura, seção 5: "Análise de Correlação — Ação se > 0.85" |
| RFE com XGBoost para seleção final | Arquitetura, seção 7: "RFE com XGBoost → ranking de importância individual" |
| Validação multi-métrica com N vs N-1 features | Extensão da arquitetura: MAE, RMSE, R², Kendall τ e top-3 accuracy |
| Anti-leakage em séries temporais | Tan et al. [18] — Instance-Conditional Timescales of Decay |

---

## Etapa 1 — Identificação e correção de leakage (25/05/2026)

Dois leakages foram identificados antes do primeiro treino de modelo. Ambos foram corrigidos antes de qualquer dado de modelo ser gerado.

### Leakage 1 — `safety_car_flag`

**O problema:** `safety_car_flag = 1` indica que *aquela corrida específica* teve Safety Car ou Virtual Safety Car. Essa informação só existe após a corrida acontecer — um modelo preditivo não sabe antes da largada se haverá SC.

**Evidência no código** (`09_preparar_base_feature_engineering.py`, linha 161):
```python
mask_sc = (
    (df["outlier_flag"] == 1)
    & (df["outlier_tipo"] == "outlier_revisao")
    & (df["safety_car_flag"] == 1)
)
```
A flag é usada para reclassificar outliers de tempo de volta — confirmando que é um dado por-corrida real (não histórico). O manifesto registra: 126 corridas com SC/VSC em 2018-2025.

**Correlação com o target:** r=-0.085 (fraca mas presente).

**Correção:** substituída por `incident_rate_hist_norm` — taxa histórica causal de SC por circuito, calculada com `expanding().mean().shift(1)` usando apenas corridas *anteriores* àquela corrida naquele circuito.

### Leakage 2 — `weather_impact_factor` (versão original)

**O problema:** a fórmula original usava temperatura, umidade e precipitação medidas *durante a corrida* via telemetria FastF1:
```
(humidity/100 + 2×rain_binary + (1 - air_temp/45)) / 4
```
Esses valores só existem enquanto a corrida acontece — um modelo pré-corrida não os conhece.

**Correlação com o target:** r=-0.013 (essencialmente zero). Uma chuva de leakage que não adicionava sinal real ao modelo.

**Correção:** recalculada como média histórica por circuito usando `expanding().mean().shift(1)`. O valor *observado* da corrida fica armazenado em `weather_impact_observed` fora de X. O RFE posterior excluiu `weather_impact_factor` do conjunto final de 13 features — o sinal histórico foi insuficiente.

---

## Etapa 2 — Análise de correlação (script `analise_correlacao_features.py`)

Após as correções de leakage, a análise identificou os pares de features com |r| > 0.85 e produziu decisões de remoção.

### Pares originais (antes das correções)

| Par | r | Ação tomada |
|---|---|---|
| `recent_form_5` × `recent_form_3` | 0.987 | `recent_form_3` **removida** — janela de 3 corridas é quase idêntica à de 5 neste dataset |
| `grid_position` × `qualifying_position` | 0.962 | `grid_position` **removida** — `qualifying_position` tem r=0.772 com target vs. 0.753 de `grid_position` |
| `recent_form_5` × `driver_constructor_synergy` | -0.874 | **Mantida** — revisão manual (ver abaixo) |

### Decisão sobre `recent_form_5` × `driver_constructor_synergy` (r=-0.874)

O arquivo `pares_correlacao_alta_maior_085.csv` registra esta decisão: `"Revisar manualmente antes de remover."`. A justificativa para manter ambas:

1. **Diferença conceitual**: `recent_form_5` captura forma geral do piloto nas últimas 5 corridas; `driver_constructor_synergy` captura o desempenho acumulado histórico do piloto com *aquela equipe específica*. Um piloto que trocou de equipe pode ter `recent_form_5` em queda mas `driver_constructor_synergy` zero (cold-start na nova equipe) — capturando a incerteza real de uma nova parceria.

2. **Evidência empírica**: ambas aparecem no top-5 de todos os modelos de árvore (documentado no documento 10), indicando que cada uma contribui com sinal independente além da outra. Se fossem puramente redundantes, o modelo usaria apenas a mais informativa.

3. **Correlação não é redundância total**: r=-0.874 significa que 24% da variância de uma não é explicada pela outra. Para features com correlação tão forte com o target (r=0.710 e r=-0.663 respectivamente), 24% de variância independente representa sinal real.

---

## Etapa 3 — RFE Temporal com XGBoost (script `rfe_xgboost_features.py`)

### Por que RFE temporal e não cross-validation padrão?

Cross-validation padrão embaralha os dados. Em séries temporais, isso cria leakage: o modelo pode treinar em 2024 e validar em 2022, "vendo o futuro". O RFE aqui usa validação temporal em três folds causais:

```
Treino: seasons ≤ 2022  →  Validação: season = 2023
Treino: seasons ≤ 2023  →  Validação: season = 2024
Treino: seasons ≤ 2024  →  Validação: season = 2025
```

As métricas medidas são erro e qualidade de ranking em folds temporais reais. Em cada fold, o modelo vê apenas temporadas anteriores à temporada de validação.

### Processo de seleção

1. Treina XGBoost com todos os candidatos em cada fold temporal.
2. Extrai ranking de importância por **gain** em cada fold e agrega por gain médio normalizado.
3. Avalia MAE, RMSE, R², Kendall τ e top-3 accuracy nos folds 2023, 2024 e 2025 para subconjuntos crescentes.
4. Normaliza as cinco métricas médias e escolhe o subconjunto com maior score composto médio.

Pesos do score composto:

| Métrica normalizada | Peso |
|---|---|
| MAE invertido | 0.30 |
| RMSE invertido | 0.15 |
| R² | 0.20 |
| Kendall τ | 0.20 |
| Top-3 accuracy | 0.15 |

### Resultado do RFE — score composto por subconjunto (do `rfe_xgboost_subsets.csv`)

| N features | Score | MAE | RMSE | R² | Kendall τ | Top-3 | Observação |
|---|---:|---:|---:|---:|---:|---:|---|
| **13** | **0.9559** | **2.3978** | **3.0727** | **0.6450** | **0.6449** | 18.7% | **Melhor compromisso global** |
| 14 | 0.4763 | 2.4143 | 3.1047 | 0.6373 | 0.6381 | **19.9%** | `incident_rate_hist_norm` melhora top-3 médio, mas piora erro/R²/ranking |
| 15 | 0.1993 | 2.4274 | 3.1030 | 0.6375 | 0.6377 | 15.7% | `driver_dnf_rate` adiciona ruído |
| 12 | 0.0971 | 2.4275 | 3.1144 | 0.6344 | 0.6308 | 18.4% | Pior equilíbrio global |

O subconjunto de 13 features permanece como melhor compromisso global. A revisão multi-fold confirmou o tamanho final, mas alterou a composição: `avg_pit_stops_circuit` entrou no X final e `incident_rate_hist_norm` ficou fora como feature direta.

### Ranking completo por gain (do `rfe_xgboost_ranking.csv`)

| Rank | Feature | Gain médio normalizado | Status |
|---|---|---|---|
| 1 | `qualifying_position` | 1.0000 | ✅ No modelo |
| 2 | `constructor_coef_rapm` | 0.2766 | ✅ No modelo |
| 3 | `recent_form_5` | 0.1695 | ✅ No modelo |
| 4 | `driver_constructor_synergy` | 0.1289 | ✅ No modelo |
| 5 | `constructor_wins_total` | 0.0431 | ✅ No modelo |
| 6 | `driver_coef_rapm` | 0.0154 | ✅ No modelo |
| 7 | `track_complexity` | 0.0140 | ✅ No modelo |
| 8 | `tire_compound_start` | 0.0116 | ✅ No modelo |
| 9 | `season_factor` | 0.0110 | ✅ No modelo |
| 10 | `avg_pit_stops_circuit` | 0.0085 | ✅ No modelo |
| 11 | `constructor_dnf_rate` | 0.0081 | ✅ No modelo |
| 12 | `grid_penalty` | 0.0074 | ✅ No modelo |
| 13 | `altitude_m` | 0.0068 | ✅ No modelo |
| **14** | **`incident_rate_hist_norm`** | 0.0038 | ❌ Excluída como feature direta |
| **15** | **`driver_dnf_rate`** | 0.0013 | ❌ Excluída |

---

## Decisão para cada feature excluída

| Feature | Motivo de exclusão | Argumento para a defesa |
|---|---|---|
| `recent_form_3` | Multicolinearidade r=0.987 com `recent_form_5` | Janela de 3 corridas é redundante — a de 5 engloba a de 3 e adiciona contexto de médio prazo |
| `grid_position` | Multicolinearidade r=0.962 com `qualifying_position` | `qualifying_position` tem correlação maior com o target (0.772 vs. 0.753) e é mais informativa por representar desempenho puro no qualifying |
| `driver_wins_total` | Excluída em rodada anterior de RFE — adição piorava MAE | Sinal capturado por `driver_coef_rapm`, que encoda histórico de sucesso de forma contínua e temporal |
| `driver_experience` | Rank 17 — pouco sinal incremental | Fortemente confundida com qualidade da equipe — pilotos de equipes top têm mais corridas *e* melhores resultados |
| `incident_rate_hist_norm` | Adição ao subconjunto de 13 piora score composto (`0.9559→0.4763`) | O sinal histórico de incidentes permanece incorporado em `track_complexity`, mas como feature direta piorou o equilíbrio médio entre erro, R² e ranking |
| `driver_dnf_rate` | Adição ao subconjunto de 14 piora score composto (`0.4763→0.1993`) | A taxa histórica de DNF do piloto não adicionou sinal incremental frente a RAPM, forma recente e sinergia piloto-construtor |
| `weather_impact_factor` | Rank 18 — sinal histórico insuficiente | Mesmo após correção de leakage, o padrão histórico de clima por circuito não é preditivo suficiente de posição final |
| `circuit_type` | Rank 19 — gain mínimo (26) | r=-0.006 com o target. Circuito urbano vs. permanente não tem poder preditivo isolado de posição final |

---

## Dataset final de modelagem

Gerado pelo script `selecao_features_modelagem.py`:

| Métrica | Valor |
|---|---|
| Linhas em X | 2.943 |
| Features em X | 13 |
| NaN em X | 0 |
| NaN em y | 0 |
| Colunas proibidas em X | 0 |
| Arquivo X | `data/processed/dataset_modelagem_X_2018_2025.csv` |
| Arquivo y | `data/processed/dataset_modelagem_y_2018_2025.csv` |
| Contrato de features | `models/feature_selection/features_modelagem_2018_2025.json` |

---

## Avaliação crítica

**Por que trocar o fold único 2025 por múltiplos folds?**

A seleção original com fold único 2025 era causal, mas sensível ao padrão de uma única temporada. A revisão executada pela Emili substituiu essa simplificação por validação temporal multi-fold em 2023, 2024 e 2025. O resultado manteve 13 features, mas tornou a seleção mais robusta e mudou a composição final: `avg_pit_stops_circuit` entrou e `incident_rate_hist_norm` saiu como feature direta.

**O par `recent_form_5` × `driver_constructor_synergy` permanece com r=-0.874:**

É o único par acima do limiar de 0.85 no dataset final. A banca pode questionar. A resposta está documentada: (1) diferença conceitual válida, (2) ambas aparecem no top-5 empiricamente, (3) 24% de variância independente com r=-0.874.

**`driver_wins_total` estava na arquitetura original e foi excluída:**

A arquitetura (seção 4) a listava como feature de piloto. O RFE mostrou que adicionar `driver_wins_total` piora o MAE. A exclusão é empiricamente fundamentada — mas deve ser documentada como divergência entre planejamento e implementação.

---

## Convergência com a literatura

| Aspecto | Alinhado | Divergente | Observação |
|---|---|---|---|
| Critério r > 0.85 para remoção | ✅ | — | Arquitetura seção 5 |
| RFE com XGBoost | ✅ | — | Arquitetura seção 7 |
| Validação multi-métrica com N vs N-1 features | ✅ | — | Extensão do critério original de MAE |
| Subconjunto de 12-15 features | ✅ | — | Arquitetura previa "12-15 variáveis finais"; resultado: 13 |
| `driver_wins_total` no modelo | — | ⚠️ | Prevista na arquitetura; excluída empiricamente pelo RFE |
| RFE com múltiplos folds | ✅ | — | Revisão robusta executada com folds 2023, 2024 e 2025 |
