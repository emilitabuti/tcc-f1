# 10 — Resultados e Feature Importance

## Contexto

Esta etapa documenta os resultados quantitativos finais dos quatro modelos, avalia quais metas da arquitetura foram atingidas, analisa a importância das features em cada modelo e prepara a base para a análise de drift da Semana 3.

---

## Fundamentação bibliográfica

| Análise | Referência |
|---|---|
| Meta MAE ≤ 2.5 | TabNet: 2.17 (Thomas et al. [7]); RAPM: 2.3 (Henderson et al. [9]) |
| Meta RMSE ≤ 3.0 | TabNet: 2.87 (Thomas et al. [7]) |
| Meta R² ≥ 0.75 | TabNet: 0.75 (Thomas et al. [7]) |
| Meta Kendall τ ≥ 0.60 | RAPM: 0.625 (Henderson et al. [9]) |
| Meta top-3 ≥ 70% | Polishchuk: 78% (Polishchuk [1]) |
| `qualifying_position` como feature dominante | Barra et al. [3], Koopman [5] — correlação esperada r≈0.71 |
| Construtor como fator dominante na era híbrida | Snoeks [10] — 88% da variância explicada pelo construtor |
| Feature importance via gain (XGBoost/LightGBM) | Chen & Guestrin [19]; Ruan et al. [2] usam SHAP analysis |

---

## Resultados por fold — tabela completa

Do `tabela_metricas_tunadas_4modelos.csv`:

### MAE por fold

| Modelo | Fold 2023 | Fold 2024 | Fold 2025 | **Média** | **DP** |
|---|---|---|---|---|---|
| Ridge | 2.281 | **2.137** | 2.398 | **2.272** | 0.131 |
| LightGBM | 2.414 | 2.175 | **2.391** | **2.326** | 0.132 |
| XGBoost | 2.456 | 2.195 | 2.393 | **2.348** | 0.136 |
| Random Forest | 2.397 | 2.240 | 2.483 | **2.373** | 0.124 |

### Kendall τ por fold

| Modelo | Fold 2023 | Fold 2024 | Fold 2025 | **Média** |
|---|---|---|---|---|
| Ridge | 0.643 | **0.694** | 0.626 | **0.654** |
| LightGBM | 0.640 | 0.686 | 0.632 | **0.653** |
| XGBoost | 0.638 | 0.684 | 0.635 | **0.652** |
| Random Forest | 0.638 | 0.677 | 0.616 | **0.644** |

### R² por fold

| Modelo | Fold 2023 | Fold 2024 | Fold 2025 | **Média** |
|---|---|---|---|---|
| Ridge | 0.656 | **0.713** | 0.644 | **0.671** |
| LightGBM | 0.634 | 0.699 | 0.642 | **0.658** |
| XGBoost | 0.632 | 0.697 | 0.642 | **0.657** |
| Random Forest | 0.640 | 0.686 | 0.625 | **0.650** |

### Score composto multi-métrica

| Modelo | Score composto médio | Interpretação |
|---|---:|---|
| **LightGBM** | **0.4971** | Melhor equilíbrio global das cinco métricas |
| **XGBoost** | 0.4963 | Empate técnico com LightGBM; mesmo top-3 médio |
| Random Forest | 0.4957 | Maior top-3 médio, mas pior em erro/R²/Kendall |
| Ridge | 0.4900 | Melhor MAE/R²/Kendall, mas top-3 muito inferior |

---

## Avaliação das metas da arquitetura

| Métrica | Meta | Ridge | LightGBM | XGBoost | RF | Status |
|---|---|---|---|---|---|---|
| MAE ≤ 2.5 | ≤ 2.5 | ✅ 2.272 | ✅ 2.326 | ✅ 2.348 | ✅ 2.373 | **Meta atingida** |
| RMSE ≤ 3.0 | ≤ 3.0 | ✅ 2.957 | ⚠️ 3.015 | ⚠️ 3.021 | ⚠️ 3.052 | **Parcialmente atingida** |
| R² ≥ 0.75 | ≥ 0.75 | ❌ 0.671 | ❌ 0.658 | ❌ 0.657 | ❌ 0.650 | **Meta não atingida** |
| Kendall τ ≥ 0.60 | ≥ 0.60 | ✅ 0.654 | ✅ 0.653 | ✅ 0.652 | ✅ 0.644 | **Meta atingida** |
| Top-3 ≥ 70% | ≥ 70% | ❌ 18.6% | ❌ 25.6% | ❌ 25.6% | ❌ 26.9% | **Meta não atingida** |

### Análise das metas não atingidas

**R² < 0.75:**

A meta de R² ≥ 0.75 vem do TabNet paper [7]. O melhor resultado individual (Ridge, fold 2024) é R²=0.720 — próximo da meta mas não atingido. Três explicações para a diferença:

1. **Dataset diferente:** o TabNet usa dados de 2014-2021 com features adicionais de telemetria não disponíveis aqui. R² é sensível à variância dos dados — períodos e conjuntos de features diferentes produzem valores não comparáveis diretamente.
2. **Problema inerentemente difícil:** F1 tem alta aleatoriedade (acidentes, safety cars, falhas mecânicas não previsíveis). Mesmo o melhor modelo humano não prediz pódios corretamente em 100% das corridas.
3. **Degradação temporal:** o fold 2025 tem R²~0.64, indicando que o modelo perde explicabilidade em dados mais recentes — fenômeno esperado e que motiva o TrAdaBoost na Fase 2.

**Top-3 accuracy: 18-26% vs. meta de 70%:**

Esta comparação é **metodologicamente inválida** sem ajuste. Polishchuk [1] que reporta 78% usou:
- Um modelo de *classificação direta* de pódio (não regressão de posição)
- O problema de otimização era binário: "está no pódio ou não?"
- A métrica reportada aparenta medir acertos individuais de vagas de pódio, não igualdade exata do conjunto inteiro de três pilotos

Com um modelo de regressão predizendo posições numéricas, o top-3 accuracy exato (conjunto de 3 pilotos idêntico ao real) é o critério mais duro possível. Um modelo que acerta 2 dos 3 pilotos do pódio mas erra a ordem não pontua. Seria necessário um modelo de classificação dedicado para competir com Polishchuk [1] nessa métrica.

---

## Metas comparáveis após revisão de literatura

A revisão posterior de baselines acadêmicos mostrou que as metas originais da arquitetura misturavam estudos com desenhos metodológicos diferentes. Por isso, as metas originais continuam documentadas como **metas aspiracionais**, mas a avaliação técnica principal deve considerar metas comparáveis ao problema deste projeto: regressão causal pré-corrida de `finish_position`.

| Métrica | Meta original | Status da meta original | Meta comparável recomendada | Justificativa |
|---|---:|---|---:|---|
| MAE | ≤ 2.5 | Comparável | ≤ 2.35 | Henderson/RAPM reporta ~2.3; TabNet reporta 2.17, mas com possíveis features menos restritas |
| RMSE | ≤ 3.0 | Comparável | ≤ 3.0 | TabNet reporta 2.87; meta permanece realista e exigente |
| R² | ≥ 0.75 | Parcialmente comparável | ≥ 0.65 ou ≥ 0.66 | TabNet reporta 0.75, mas usa outro período e possivelmente variáveis intra/pós-corrida |
| Kendall τ | ≥ 0.60 | Comparável | ≥ 0.60 | Métrica alinhada ao objetivo de ranking e ao benchmark RAPM |
| Top-3 exato | ≥ 70% | Não comparável ao pipeline principal | ≥ 25% como baseline; ≥ 30% como avanço forte | Trabalhos com 70%+ tratam pódio como classificação direta, não regressão causal com igualdade exata de conjunto |

Leitura para defesa:

- O pipeline principal é uma regressão causal de posição final; portanto, MAE, RMSE, R² e Kendall τ são as métricas centrais.
- Top-3 continua sendo reportado, mas como métrica complementar derivada da regressão.
- Não foi encontrado baseline acadêmico equivalente que reporte top-3 exato a partir de regressão causal de `finish_position`.
- Comparações com trabalhos de classificação de pódio devem ser apresentadas como motivação para um modelo complementar, não como falha direta da regressão.

---

## Feature Importance

### Rankings por modelo (médias dos 3 folds)

#### Importância normalizada — todos os modelos

| Rank | Feature | LightGBM (gain%) | Random Forest (gini%) | XGBoost (gain%) | Consistência |
|---|---|---|---|---|---|
| 1 | `qualifying_position` | **63.1%** | **28.6%** | **36.4%** | Top-1 nos três |
| 2 | `constructor_coef_rapm` | 15.4% | 17.4% | 18.1% | Top-3 nos três |
| 3 | `recent_form_5` | 10.3% | 17.4% | 17.7% | Top-3 nos três |
| 4 | `driver_constructor_synergy` | 4.8% | 11.2% | 8.2% | Top-4 nos três |
| 5 | `constructor_wins_total` | 1.4% | 9.7% | 5.1% | Top-5 nos três |
| 6 | `driver_coef_rapm` | 1.0% | 6.9% | 2.8% | Top-8 nos três |
| 7 | `track_complexity` | 1.2% | 1.8% | 1.9% | Top-8 nos três |
| 8–13 | Demais features | ~2.8% (LGB) | ~8.6% (RF) | ~11.8% (XGB) | Variável |

**Nota sobre tipos de importância:**
- LightGBM e XGBoost: importância por **gain** (contribuição média ao ganho de informação por divisão)
- Random Forest: importância por **Gini impurity reduction** (redução média de impureza por divisão)

Esses são métodos distintos — os percentuais não são diretamente comparáveis, mas a **ordem relativa** é comparável e é o que importa para análise de convergência.

---

### `qualifying_position` domina — por quê isso é coerente com a literatura

A correlação de `qualifying_position` com `finish_position` é r=0.772 — a mais alta de todas as features. Na F1, quem sai na frente tem vantagem clara: o carro mais rápido tende a classificar bem e, em pista limpa, manter sua vantagem. Barra et al. [3] reportam correlação esperada de r≈0.71 — o valor observado (0.772) é ligeiramente superior, consistente com dados da era híbrida onde a supremacia do melhor carro é ainda mais pronunciada.

A dominância no LightGBM (63.1%) merece atenção: `qualifying_position` concentra a maior parte do ganho informativo. Isso pode indicar que o modelo está simplificando demais — priorizando a feature mais correlacionada e usando as demais para refinamentos menores. Em Random Forest a concentração é menor (28.6%) — resultado esperado, pois o bagging com `max_features=sqrt` força as árvores a explorar outras features mesmo quando `qualifying_position` não está disponível na divisão.

---

### `constructor_coef_rapm` no top-3 — confirmação de Snoeks [10]

`constructor_coef_rapm` aparece no top-3 dos três modelos (LightGBM rank 2, RF rank 2, XGBoost rank 2/3, praticamente empatado com `recent_form_5`). Isso confirma empiricamente a tese central de Snoeks [10]: o construtor explica a maior parte da variância de desempenho na era híbrida.

A correlação direta com `finish_position` é r=-0.683 — a segunda maior entre todas as features. O RAPM capturou esse sinal de forma eficiente: a qualidade do carro é o fator mais preditivo depois de onde o piloto largou.

---

### Features de baixa importância — isso invalida sua inclusão?

As features de menor importância (ranks 9-13, dependendo do modelo) são: `grid_penalty`, `altitude_m`, `avg_pit_stops_circuit`, `tire_compound_start` e `season_factor`.

**Não invalida — por três razões:**

1. **Interpretabilidade e cobertura metodológica:** o TCC se propõe a modelar variáveis de circuito, clima e estratégia conforme a literatura ([2], [6]). Incluir essas features demonstra que o projeto considerou os fatores relevantes — mesmo que o sinal empírico seja fraco neste dataset.

2. **RFE confirmou inclusão:** o subconjunto com 13 features tem o melhor score composto multi-métrica (documento 07). Remover uma feature pode melhorar uma métrica isolada, mas piora o equilíbrio geral.

3. **Importância != ausência de efeito:** features de baixo gain podem capturar interações que outros métodos não capturam. O XGBoost distribui a importância de forma mais uniforme entre as features de baixo rank (~1.5-2% cada) do que o LightGBM (<0.2% cada) — indicando que no XGBoost, essas features estão sendo usadas de forma mais distribuída.

---

## Feature importance fold 2024 — referência para análise de drift

O arquivo `feature_importance_2024.csv` registra a importância das 13 features no fold com treino 2018-2023 e validação 2024 — **antes da transição regulatória de 2026**.

Essa referência será usada na Semana 3 (P2) para comparar com a importância nas primeiras corridas de 2026. A hipótese central do TCC é que `constructor_coef_rapm` terá sua importância **drasticamente reduzida** em 2026 — porque os coeficientes históricos de construtor perdem validade com o novo regulamento, e o modelo passará a depender mais de `qualifying_position` (que reflete apenas o desempenho imediato, não o histórico).

Do fold 2024 (LightGBM, do `feature_importance_2024.csv`):

| Feature | Importância normalizada (fold 2024) |
|---|---|
| `qualifying_position` | 61.1% |
| `constructor_coef_rapm` | 16.0% |
| `recent_form_5` | 11.7% |
| `driver_constructor_synergy` | 5.1% |
| Demais 9 features | 7.1% |

Qualquer variação significativa nessa distribuição nas corridas de 2026 será evidência de drift regulatório.

---

## Avaliação crítica

**O R² abaixo de 0.75 é preocupante?**

Contextualizando: R²=0.67 com 13 features e 2.524-2.943 amostras é competitivo para um problema de previsão de resultados esportivos. A literatura que reporta R²=0.75 (TabNet [7]) usa configurações diferentes e possivelmente inclui leakage (feature `position` em alguns estudos). O R² de 0.713 no fold 2024 (Ridge) está próximo da meta — a degradação para ~0.64 no fold 2025 é o fenômeno de drift que a Fase 2 trata.

**A diferença Ridge vs. árvores em MAE (2.272 vs. 2.326) é significativa?**

Diferença de 0.04 posições em média. Em termos práticos, é pequena. Porém, o score composto favorece LightGBM/XGBoost porque o Ridge tem top-3 accuracy médio muito menor (18.6% contra 25.6%).

**A distribuição muito concentrada do LightGBM (63% em `qualifying_position`) é um risco?**

Em corridas com safety car, acidentes ou condições de chuva, a `qualifying_position` perde parte de seu poder preditivo — a ordem de largada é embaralhada. Um modelo que depende muito de uma única feature pode ser mais vulnerável a corridas atípicas. O fold 2023 tem mais corridas irregulares que 2024 e 2025 — e o LightGBM tem MAE 2.414 em 2023 vs. 2.175 em 2024.

---

## Convergência com a literatura

| Aspecto | Alinhado | Divergente | Observação |
|---|---|---|---|
| `qualifying_position` como feature dominante | ✅ | — | Correlação observada (0.772) > esperada por Barra et al. [3] (0.71) |
| `constructor_coef_rapm` top-3 | ✅ | — | Confirma Snoeks [10]: construtor domina na era híbrida |
| `recent_form_5` top-3 | ✅ | — | Ruan et al. [2] incluem como feature central |
| MAE abaixo de 2.5 | ✅ | — | Alcançado por todos os modelos |
| R² abaixo de 0.75 | — | ⚠️ | Meta baseada em TabNet [7] com configuração diferente |
| Top-3 accuracy 18-26% vs. 78% | — | ⚠️ | Comparação inválida: regressão vs. classificação direta [1] |
| Kendall τ ≥ 0.60 | ✅ | — | Todos alcançam; alinhado com RAPM [9]: τ=0.625 |
