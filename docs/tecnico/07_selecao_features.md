# 07 — Seleção de Features

## Contexto

Com todas as features criadas, o conjunto original era de 19 candidatas. O processo de seleção tem dois objetivos: (1) remover features que introduzem leakage ou multicolinearidade severa; (2) encontrar o subconjunto que maximiza a performance preditiva no conjunto de validação temporal.

O processo seguiu três etapas em sequência:

1. **Identificação e correção de leakage** — remoção de dados pós-corrida (25/05/2026).
2. **Análise de correlação** — remoção de redundâncias severas (r > 0.85).
3. **RFE temporal com XGBoost** — seleção do subconjunto ótimo por MAE no fold 2025.

---

## Fundamentação bibliográfica

| Decisão | Referência |
|---|---|
| Critério r > 0.85 para multicolinearidade | Arquitetura, seção 5: "Análise de Correlação — Ação se > 0.85" |
| RFE com XGBoost para seleção final | Arquitetura, seção 7: "RFE com XGBoost → ranking de importância individual" |
| Validação MAE com N vs N-1 features | Arquitetura, seção 7: "Validação: comparar MAE com N features vs N-1 features" |
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

**Correção:** recalculada como média histórica por circuito usando `expanding().mean().shift(1)`. O valor *observado* da corrida fica armazenado em `weather_impact_observed` fora de X. O RFE posterior excluiu `weather_impact_factor` do conjunto final de 15 features — o sinal histórico foi insuficiente.

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

Cross-validation padrão embaralha os dados. Em séries temporais, isso cria leakage: o modelo pode treinar em 2024 e validar em 2022, "vendo o futuro". O RFE aqui usa um único split temporal:

```
Treino: seasons ≤ 2024  →  2524 linhas
Validação: season = 2025  →  419 linhas
```

O MAE medido é o erro real de predição — o modelo nunca viu dados de 2025 durante o processo de seleção.

### Processo de seleção

1. Treina XGBoost com todos os candidatos nas mesmas configurações do walk-forward.
2. Extrai ranking de importância por **gain** (contribuição média ao ganho de informação).
3. Avalia o MAE no fold 2025 para subconjuntos crescentes: top-1, top-2, ..., top-19 features.
4. Escolhe o subconjunto com menor MAE.

### Resultado do RFE — MAE por subconjunto (do `rfe_xgboost_subsets.csv`)

| N features | MAE fold 2025 | Observação |
|---|---|---|
| 12 | 2.5099 | Abaixo do mínimo |
| 13 | 2.4921 | Melhorando |
| 14 | 2.4781 | Melhorando |
| **15** | **2.4669** | **Mínimo global — subconjunto escolhido** |
| 16 | 2.4938 | Piora — `driver_wins_total` adicionada |
| 17 | 2.5621 | Continua piorando |
| 18 | 2.5392 | Piora |
| 19 | 2.5140 | Piora |

O subconjunto de 15 features é o mínimo global. A adição da 16ª feature (`driver_wins_total`) já aumenta o MAE — indica que a feature adiciona ruído em vez de sinal.

### Ranking completo por gain (do `rfe_xgboost_ranking.csv`)

| Rank | Feature | Gain | Status |
|---|---|---|---|
| 1 | `qualifying_position` | 1.096 | ✅ No modelo |
| 2 | `recent_form_5` | 250 | ✅ No modelo |
| 3 | `constructor_coef_rapm` | 222 | ✅ No modelo |
| 4 | `driver_constructor_synergy` | 146 | ✅ No modelo |
| 5 | `constructor_wins_total` | 109 | ✅ No modelo |
| 6 | `season_factor` | 56 | ✅ No modelo |
| 7 | `tire_compound_start` | 53 | ✅ No modelo |
| 8 | `driver_coef_rapm` | 43 | ✅ No modelo |
| 9 | `incident_rate_hist_norm` | 41 | ✅ No modelo |
| 10 | `altitude_m` | 39 | ✅ No modelo |
| 11 | `avg_pit_stops_circuit` | 37 | ✅ No modelo |
| 12 | `track_complexity` | 36 | ✅ No modelo |
| 13 | `constructor_dnf_rate` | 36 | ✅ No modelo |
| 14 | `grid_penalty` | 35 | ✅ No modelo |
| 15 | `driver_dnf_rate` | 33 | ✅ No modelo |
| **16** | **`driver_wins_total`** | 32 | ❌ Excluída — piora o MAE |
| **17** | **`driver_experience`** | 32 | ❌ Excluída |
| **18** | **`weather_impact_factor`** | 29 | ❌ Excluída |
| **19** | **`circuit_type`** | 26 | ❌ Excluída |

---

## Decisão para cada feature excluída

| Feature | Motivo de exclusão | Argumento para a defesa |
|---|---|---|
| `recent_form_3` | Multicolinearidade r=0.987 com `recent_form_5` | Janela de 3 corridas é redundante — a de 5 engloba a de 3 e adiciona contexto de médio prazo |
| `grid_position` | Multicolinearidade r=0.962 com `qualifying_position` | `qualifying_position` tem correlação maior com o target (0.772 vs. 0.753) e é mais informativa por representar desempenho puro no qualifying |
| `driver_wins_total` | Rank 16 — adição piora MAE (2.4669→2.4938) | Sinal capturado por `driver_coef_rapm`, que encoda histórico de sucesso de forma contínua e temporal |
| `driver_experience` | Rank 17 — pouco sinal incremental | Fortemente confundida com qualidade da equipe — pilotos de equipes top têm mais corridas *e* melhores resultados |
| `weather_impact_factor` | Rank 18 — sinal histórico insuficiente | Mesmo após correção de leakage, o padrão histórico de clima por circuito não é preditivo suficiente de posição final |
| `circuit_type` | Rank 19 — gain mínimo (26) | r=-0.006 com o target. Circuito urbano vs. permanente não tem poder preditivo isolado de posição final |

---

## Dataset final de modelagem

Gerado pelo script `selecao_features_modelagem.py`:

| Métrica | Valor |
|---|---|
| Linhas em X | 2.943 |
| Features em X | 15 |
| NaN em X | 0 |
| NaN em y | 0 |
| Colunas proibidas em X | 0 |
| Arquivo X | `data/processed/dataset_modelagem_X_2018_2025.csv` |
| Arquivo y | `data/processed/dataset_modelagem_y_2018_2025.csv` |
| Contrato de features | `models/feature_selection/features_modelagem_2018_2025.json` |

---

## Avaliação crítica

**O RFE com um único fold 2025 é suficiente?**

Usar apenas o fold 2025 para seleção de features é uma simplificação. Idealmente, o RFE seria executado com cross-validation temporal em múltiplos folds (2023, 2024, 2025) e o subconjunto selecionado seria aquele com menor MAE médio. Com um único fold, existe o risco de que o subconjunto de 15 features esteja ligeiramente sobreajustado ao padrão de 2025. Para a defesa: o fold 2025 foi escolhido por ser o mais recente e, portanto, o mais representativo do desempenho esperado em 2026. Além disso, o RFE foi executado após todos os hiperparâmetros de limpeza e feature engineering terem sido fixados — sem iterações de olhar o resultado e ajustar o pipeline.

**O par `recent_form_5` × `driver_constructor_synergy` permanece com r=-0.874:**

É o único par acima do limiar de 0.85 no dataset final. A banca pode questionar. A resposta está documentada: (1) diferença conceitual válida, (2) ambas aparecem no top-5 empiricamente, (3) 24% de variância independente com r=-0.874.

**`driver_wins_total` estava na arquitetura original e foi excluída:**

A arquitetura (seção 4) a listava como feature de piloto. O RFE mostrou que adicionar `driver_wins_total` ao conjunto de 15 piora o MAE. A exclusão é empiricamente fundamentada — mas deve ser documentada como divergência entre planejamento e implementação.

---

## Convergência com a literatura

| Aspecto | Alinhado | Divergente | Observação |
|---|---|---|---|
| Critério r > 0.85 para remoção | ✅ | — | Arquitetura seção 5 |
| RFE com XGBoost | ✅ | — | Arquitetura seção 7 |
| Validação por MAE com N vs N-1 features | ✅ | — | Arquitetura seção 7 |
| Subconjunto de 12-15 features | ✅ | — | Arquitetura previa "12-15 variáveis finais"; resultado: 15 |
| `driver_wins_total` no modelo | — | ⚠️ | Prevista na arquitetura; excluída empiricamente pelo RFE |
| RFE com múltiplos folds | — | ⚠️ | Executado com fold único 2025; arquitetura não especificava |
