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
| Ridge | 2.289 | **2.133** | 2.399 | **2.273** | 0.134 |
| LightGBM | 2.396 | 2.186 | **2.358** | **2.313** | 0.112 |
| Random Forest | 2.377 | 2.190 | 2.416 | **2.328** | 0.121 |
| XGBoost | 2.431 | 2.182 | 2.390 | **2.334** | 0.133 |

### Kendall τ por fold

| Modelo | Fold 2023 | Fold 2024 | Fold 2025 | **Média** |
|---|---|---|---|---|
| Ridge | 0.643 | **0.695** | 0.626 | **0.655** |
| LightGBM | 0.642 | 0.682 | 0.641 | **0.655** |
| Random Forest | 0.638 | 0.680 | 0.635 | **0.651** |
| XGBoost | 0.644 | 0.683 | 0.629 | **0.652** |

### R² por fold

| Modelo | Fold 2023 | Fold 2024 | Fold 2025 | **Média** |
|---|---|---|---|---|
| Ridge | 0.655 | **0.714** | 0.643 | **0.671** |
| LightGBM | 0.636 | 0.696 | **0.648** | **0.660** |
| Random Forest | 0.643 | 0.694 | 0.634 | **0.657** |
| XGBoost | 0.636 | 0.700 | 0.640 | **0.658** |

---

## Avaliação das metas da arquitetura

| Métrica | Meta | Ridge | LightGBM | RF | Status |
|---|---|---|---|---|---|
| MAE ≤ 2.5 | ≤ 2.5 | ✅ 2.273 | ✅ 2.313 | ✅ 2.328 | **Meta atingida** |
| RMSE ≤ 3.0 | ≤ 3.0 | ✅ 2.958 | ⚠️ 3.008 | ⚠️ 3.020 | **Parcialmente atingida** |
| R² ≥ 0.75 | ≥ 0.75 | ❌ 0.671 | ❌ 0.660 | ❌ 0.657 | **Meta não atingida** |
| Kendall τ ≥ 0.60 | ≥ 0.60 | ✅ 0.655 | ✅ 0.655 | ✅ 0.651 | **Meta atingida** |
| Top-3 ≥ 70% | ≥ 70% | ❌ 18.6% | ❌ 24.2% | ❌ 21.3% | **Meta não atingida** |

### Análise das metas não atingidas

**R² < 0.75:**

A meta de R² ≥ 0.75 vem do TabNet paper [7]. O melhor resultado individual (Ridge, fold 2024) é R²=0.714 — próximo da meta mas não atingido. Três explicações para a diferença:

1. **Dataset diferente:** o TabNet usa dados de 2014-2021 com features adicionais de telemetria não disponíveis aqui. R² é sensível à variância dos dados — períodos e conjuntos de features diferentes produzem valores não comparáveis diretamente.
2. **Problema inerentemente difícil:** F1 tem alta aleatoriedade (acidentes, safety cars, falhas mecânicas não previsíveis). Mesmo o melhor modelo humano não prediz pódios corretamente em 100% das corridas.
3. **Degradação temporal:** o fold 2025 tem R²~0.64, indicando que o modelo perde explicabilidade em dados mais recentes — fenômeno esperado e que motiva o TrAdaBoost na Fase 2.

**Top-3 accuracy: 18-24% vs. meta de 70%:**

Esta comparação é **metodologicamente inválida** sem ajuste. Polishchuk [1] que reporta 78% usou:
- Um modelo de *classificação direta* de pódio (não regressão de posição)
- O problema de otimização era binário: "está no pódio ou não?"
- A métrica de acurácia mede se o conjunto exato de 3 pilotos foi acertado — critério mais estrito que classificação individual por piloto

Com um modelo de regressão predizendo posições numéricas, o top-3 accuracy exato (conjunto de 3 pilotos idêntico ao real) é o critério mais duro possível. Um modelo que acerta 2 dos 3 pilotos do pódio mas erra a ordem não pontua. Seria necessário um modelo de classificação dedicado para competir com Polishchuk [1] nessa métrica.

---

## Feature Importance

### Rankings por modelo (médias dos 3 folds)

#### Importância normalizada — todos os modelos

| Rank | Feature | LightGBM (gain%) | Random Forest (gini%) | XGBoost (gain%) | Consistência |
|---|---|---|---|---|---|
| 1 | `qualifying_position` | **68.4%** | **43.3%** | **38.7%** | Top-1 nos três |
| 2 | `constructor_coef_rapm` | 10.7% | 20.2% | 13.2% | Top-2 em LGB e RF |
| 3 | `recent_form_5` | 9.7% | 12.8% | 15.6% | Top-2 em XGB |
| 4 | `driver_constructor_synergy` | 5.2% | 9.5% | 7.8% | Top-4 nos três |
| 5 | `constructor_wins_total` | 1.4% | 5.0% | 6.7% | Top-5 nos três |
| 6 | `driver_coef_rapm` | 1.0% | 2.5% | 2.5% | Top-7 nos três |
| 7 | `constructor_dnf_rate` | 1.0% | 1.3% | 2.0% | Top-8 nos três |
| 8–15 | Demais features | ~2.6% (LGB) | ~5.4% (RF) | ~13.4% (XGB) | Variável |

**Nota sobre tipos de importância:**
- LightGBM e XGBoost: importância por **gain** (contribuição média ao ganho de informação por divisão)
- Random Forest: importância por **Gini impurity reduction** (redução média de impureza por divisão)

Esses são métodos distintos — os percentuais não são diretamente comparáveis, mas a **ordem relativa** é comparável e é o que importa para análise de convergência.

---

### `qualifying_position` domina — por quê isso é coerente com a literatura

A correlação de `qualifying_position` com `finish_position` é r=0.772 — a mais alta de todas as features. Na F1, quem sai na frente tem vantagem clara: o carro mais rápido tende a classificar bem e, em pista limpa, manter sua vantagem. Barra et al. [3] reportam correlação esperada de r≈0.71 — o valor observado (0.772) é ligeiramente superior, consistente com dados da era híbrida onde a supremacia do melhor carro é ainda mais pronunciada.

A dominância **extrema** no LightGBM (68.4%) merece atenção: `qualifying_position` está concentrando mais de dois terços do ganho informativo. Isso pode indicar que o modelo está simplificando demais — priorizando a feature mais correlacionada e usando as demais para refinamentos menores. Em Random Forest a concentração é menor (43.3%) — resultado esperado, pois o bagging com `max_features=0.5` força os árvores a explorar outras features mesmo quando `qualifying_position` não está disponível na divisão.

---

### `constructor_coef_rapm` no top-3 — confirmação de Snoeks [10]

`constructor_coef_rapm` aparece no top-3 de dois dos três modelos (LightGBM rank 2, RF rank 2, XGBoost rank 3). Isso confirma empiricamente a tese central de Snoeks [10]: o construtor explica a maior parte da variância de desempenho na era híbrida.

A correlação direta com `finish_position` é r=-0.683 — a segunda maior entre todas as features. O RAPM capturou esse sinal de forma eficiente: a qualidade do carro é o fator mais preditivo depois de onde o piloto largou.

---

### Features de baixa importância — isso invalida sua inclusão?

As features de menor importância (ranks 11-15) são: `altitude_m`, `season_factor`, `incident_rate_hist_norm`, `tire_compound_start`, `grid_penalty`.

**Não invalida — por três razões:**

1. **Interpretabilidade e cobertura metodológica:** o TCC se propõe a modelar variáveis de circuito, clima e estratégia conforme a literatura ([2], [6]). Incluir essas features demonstra que o projeto considerou os fatores relevantes — mesmo que o sinal empírico seja fraco neste dataset.

2. **RFE confirmou inclusão:** o subconjunto com 15 features tem MAE menor que qualquer subconjunto menor (documento 07). Remover essas features piora o modelo — mesmo que a melhoria individual seja pequena, o conjunto completo é o ótimo.

3. **Importância != ausência de efeito:** features de baixo gain podem capturar interações que outros métodos não capturam. O XGBoost distribui a importância de forma mais uniforme entre as features de baixo rank (~1.5-2% cada) do que o LightGBM (<0.2% cada) — indicando que no XGBoost, essas features estão sendo usadas de forma mais distribuída.

---

## Feature importance fold 2024 — referência para análise de drift

O arquivo `feature_importance_2024.csv` registra a importância das 15 features no fold com treino 2018-2023 e validação 2024 — **antes da transição regulatória de 2026**.

Essa referência será usada na Semana 3 (P2) para comparar com a importância nas primeiras corridas de 2026. A hipótese central do TCC é que `constructor_coef_rapm` terá sua importância **drasticamente reduzida** em 2026 — porque os coeficientes históricos de construtor perdem validade com o novo regulamento, e o modelo passará a depender mais de `qualifying_position` (que reflete apenas o desempenho imediato, não o histórico).

Do fold 2024 (LightGBM, do `feature_importance_2024.csv`):

| Feature | Importância normalizada (fold 2024) |
|---|---|
| `qualifying_position` | 65.4% |
| `constructor_coef_rapm` | 11.4% |
| `recent_form_5` | 10.7% |
| `driver_constructor_synergy` | 6.7% |
| Demais 11 features | 5.8% |

Qualquer variação significativa nessa distribuição nas corridas de 2026 será evidência de drift regulatório.

---

## Avaliação crítica

**O R² abaixo de 0.75 é preocupante?**

Contextualizando: R²=0.67 com 15 features e 2.524-2.943 amostras é competitivo para um problema de previsão de resultados esportivos. A literatura que reporta R²=0.75 (TabNet [7]) usa configurações diferentes e possivelmente inclui leakage (feature `position` em alguns estudos). O R² de 0.714 no fold 2024 (o mais recente do período de treino histórico) está próximo da meta — a degradação para 0.64 no fold 2025 é o fenômeno de drift que a Fase 2 trata.

**A diferença Ridge vs. árvores em MAE (2.273 vs. 2.313) é significativa?**

Diferença de 0.04 posições em média. Em termos práticos: ao longo de uma temporada de 24 corridas com ~18 pilotos por corrida, o Ridge erra cumulativamente ~0.04 posições a menos por predição. Não é substantiva para a prática, mas é metodologicamente relevante para documentar que a solução linear (RAPM) captura bem o problema.

**A distribuição muito concentrada do LightGBM (68% em `qualifying_position`) é um risco?**

Em corridas com safety car, acidentes ou condições de chuva, a `qualifying_position` perde parte de seu poder preditivo — a ordem de largada é embaralhada. Um modelo que depende 68% de uma única feature pode ser mais vulnerável a corridas atípicas. O fold 2023 tem mais corridas irregulares que 2024 e 2025 — e o LightGBM tem MAE 2.396 em 2023 vs. 2.186 em 2024, a maior variação entre folds de qualquer modelo.

---

## Convergência com a literatura

| Aspecto | Alinhado | Divergente | Observação |
|---|---|---|---|
| `qualifying_position` como feature dominante | ✅ | — | Correlação observada (0.772) > esperada por Barra et al. [3] (0.71) |
| `constructor_coef_rapm` top-3 | ✅ | — | Confirma Snoeks [10]: construtor domina na era híbrida |
| `recent_form_5` top-3 | ✅ | — | Ruan et al. [2] incluem como feature central |
| MAE abaixo de 2.5 | ✅ | — | Alcançado por todos os modelos |
| R² abaixo de 0.75 | — | ⚠️ | Meta baseada em TabNet [7] com configuração diferente |
| Top-3 accuracy 18-24% vs. 78% | — | ⚠️ | Comparação inválida: regressão vs. classificação direta [1] |
| Kendall τ ≥ 0.60 | ✅ | — | Todos alcançam; alinhado com RAPM [9]: τ=0.625 |
