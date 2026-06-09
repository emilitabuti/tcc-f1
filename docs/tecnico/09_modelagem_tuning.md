# 09 — Modelagem e Tuning

## Contexto

Com o dataset de modelagem pronto e a infraestrutura de walk-forward estabelecida, o objetivo desta etapa é treinar e otimizar quatro modelos — três de árvore e um baseline linear — e selecionar os dois finalistas para a apresentação da Fase 1.

---

## Fundamentação bibliográfica

| Algoritmo | Referência principal |
|---|---|
| XGBoost | Chen & Guestrin [19] — referência original |
| Random Forest | Breiman [20] — referência original |
| LightGBM | Ke et al. (2017); motivação empírica: Barra et al. [3] reportam R²=0.999 com LightGBM em F1 |
| Ridge baseline | Henderson et al. [9] — "Adaptado diretamente do RAPM paper. Serve como referência simples para mostrar que os modelos de árvore realmente agregam valor em relação a uma solução linear." |
| Optuna (Bayesian optimization) | Akiba et al. [23] — "Optuna: A Next-generation Hyperparameter Optimization Framework" |
| Justificativa teórica de busca de hiperparâmetros | Bergstra & Bengio [22] — Random Search for Hyper-Parameter Optimization |

---

## Os quatro modelos

### XGBoost — Gradient Boosting sequencial

**Filosofia:** constrói árvores em sequência. Cada nova árvore é treinada nos resíduos (erros) da anterior — *corrigindo o que o modelo anterior errou*. Resultado: cada árvore é fraca individualmente, mas o conjunto converge para um modelo forte.

**Regularização:** L1 (`reg_alpha`) e L2 (`reg_lambda`) nativos — controlam a complexidade dos coeficientes das folhas.

**Crescimento level-wise:** XGBoost cresce todas as folhas de um nível antes de passar ao próximo — árvores balanceadas.

**Referência:** Chen & Guestrin [19].

---

### LightGBM — Gradient Boosting leaf-wise

**Filosofia:** mesma base do XGBoost (boosting sequencial), mas com duas diferenças técnicas:

1. **Crescimento leaf-wise:** em vez de crescer nível por nível, LightGBM cresce a folha com maior ganho de impureza. Resultado: árvores assimétricas que podem ser mais precisas com menos profundidade.

2. **GOSS (Gradient-based One-Side Sampling):** ao invés de usar todas as amostras de treino, amostra mais agressivamente os exemplos com gradiente pequeno (fáceis) e mantém todos os com gradiente grande (difíceis). Acelera o treinamento sem perda significativa de precisão.

**`num_leaves` vs. `max_depth`:** LightGBM controla a complexidade principalmente via `num_leaves` (número máximo de folhas), não apenas `max_depth`. Para a mesma `max_depth`, `num_leaves` menor produz árvores mais simples.

**Referência:** motivação Barra et al. [3] — porém atenção: o R²=0.999 reportado por Barra et al. usa `finish_position` como feature de entrada, o que configura leakage confirmado. O cronograma (seção "Pontos de Atenção") documenta isso explicitamente: "Para o nosso setup genuíno, espera-se MAE próximo ao XGBoost."

---

### Random Forest — Bagging paralelo

**Filosofia:** constrói árvores *independentemente em paralelo*. Cada árvore usa um subconjunto aleatório de dados (bootstrap) e de features (`max_features`). O resultado final é a média das predições individuais.

**Regularização:** implícita pelo ensemble — a média de muitas árvores reduz variância naturalmente.

**Comportamento em drift:** por ser uma média de árvores independentes, degrada mais suavemente com mudanças de distribuição — expectativa teórica relevante para 2026.

**Referência:** Breiman [20].

---

### Ridge Regression — Baseline linear

**Filosofia:** regressão linear com penalização L2 nos coeficientes. Adaptado do RAPM paper de Henderson et al. [9] — serve para evidenciar que os coeficientes lineares de piloto e construtor capturam grande parte do sinal preditivo.

**Normalização:** `StandardScaler` ajustado no treino de cada fold antes de aplicar ao modelo — evita leakage de escala entre folds.

**Referência:** Henderson et al. [9].

---

## Optuna — Processo de tuning

**Script:** cada modelo tem um script dedicado (`tuning_xgboost.py`, `tuning_randomforest.py`, `tuning_lightgbm.py`) que usa `tuning_utils.py` como módulo compartilhado.

### Por que Optuna e não Grid Search?

Grid Search testa todas as combinações de uma grade pré-definida. Com 7 hiperparâmetros e 5 valores cada, seriam 5⁷ = 78.125 combinações. Optuna usa o **algoritmo TPE** (*Tree-structured Parzen Estimator*) — um método bayesiano que aprende quais regiões do espaço de busca são promissoras e concentra os trials nessas regiões. Com 50 trials, o TPE é geralmente mais eficiente que Random Search [22] e muito mais que Grid Search.

**Seed fixo:** `TPESampler(seed=42)` — reprodutibilidade garantida.

**Objetivo revisado:** após a revisão de seleção de features, o tuning passou a maximizar um score composto multi-métrica nos folds 2023-2024. O score combina MAE invertido (0.30), RMSE invertido (0.15), R² (0.20), Kendall τ (0.20) e top-3 accuracy (0.15). Isso evita escolher hiperparâmetros bons apenas em MAE, mas ruins para ranking ou pódio.

### Folds de tuning vs. folds de avaliação

**Folds de tuning** (definidos em `tuning_utils.py`): apenas folds 2023 e 2024.
```
FOLDS_TUNING = [
    {"train_until": 2022, "valid_season": 2023},
    {"train_until": 2023, "valid_season": 2024},
]
```

**Folds de avaliação final**: todos os três folds, incluindo 2025.
```
FOLDS_AVALIACAO = [
    {"train_until": 2022, "valid_season": 2023},
    {"train_until": 2023, "valid_season": 2024},
    {"train_until": 2024, "valid_season": 2025},  ← fold 2025 nunca visto no tuning
]
```

O fold 2025 é reservado como holdout para o tuning — os hiperparâmetros são escolhidos sem qualquer acesso ao desempenho em 2025. A seleção de features, documentada no arquivo 07, usa 2023-2025 em RFE temporal multi-fold; portanto, 2025 não é holdout absoluto do pipeline inteiro, mas permanece isolado da otimização de hiperparâmetros.

### Espaços de busca por modelo

**XGBoost** (7 parâmetros, script `tuning_xgboost.py`):

| Parâmetro | Faixa | Escala |
|---|---|---|
| `n_estimators` | 100–500 | Inteiro |
| `max_depth` | 3–10 | Inteiro |
| `learning_rate` | 0.01–0.30 | Log |
| `subsample` | 0.60–1.00 | Float |
| `colsample_bytree` | 0.60–1.00 | Float |
| `reg_alpha` | 0.00–1.00 | Float |
| `reg_lambda` | 0.00–1.00 | Float |

**LightGBM** (9 parâmetros, inclui `num_leaves` e `min_child_samples`):

| Parâmetro | Faixa | Observação |
|---|---|---|
| `n_estimators` | 100–500 | Mesmo range que XGBoost |
| `max_depth` | 3–10 | Igual |
| `num_leaves` | 7–min(127, 2^max_depth-1) | Constraindo para `num_leaves ≤ 2^max_depth - 1` |
| `learning_rate` | 0.01–0.30 | Log |
| `subsample` | 0.60–1.00 | |
| `colsample_bytree` | 0.60–1.00 | |
| `reg_alpha` | 0.00–1.00 | |
| `reg_lambda` | 0.00–1.00 | |
| `min_child_samples` | 5–40 | Equivalente ao `min_samples_leaf` do RF |

**Random Forest** (5 parâmetros):

| Parâmetro | Faixa | Observação |
|---|---|---|
| `n_estimators` | 100–500 | |
| `max_depth` | 3–15 | Maior que XGBoost/LGB (RF tolera mais profundidade) |
| `max_features` | sqrt, log2, 0.5 | Candidatos discretos |
| `min_samples_split` | 2–10 | |
| `min_samples_leaf` | 1–5 | |

**Ridge** (grid-search temporal, não Optuna):

Varre `alpha` em escala log de 0.01 a 100 com `cross_val` temporal. Resultado: `alpha=0.01` — regularização mínima. Isso significa que os coeficientes lineares têm alta liberdade, o que é coerente com a força do sinal linear (construtor domina a variância).

---

## Hiperparâmetros ótimos encontrados

| Modelo | Parâmetros ótimos | Tempo de tuning |
|---|---|---|
| **XGBoost** | `n_estimators=344, max_depth=3, lr=0.010, subsample=0.952, colsample_bytree=0.686, reg_alpha=0.039, reg_lambda=0.913` | 36.7s |
| **LightGBM** | `n_estimators=230, max_depth=3, num_leaves=7, lr=0.016, subsample=0.862, colsample_bytree=0.769, reg_alpha=0.659, reg_lambda=0.078, min_child_samples=22` | 22.0s |
| **Random Forest** | `n_estimators=231, max_depth=11, max_features=log2, min_samples_split=3, min_samples_leaf=5` | 68.7s |
| **Ridge** | `alpha=0.01` com `StandardScaler` | 4.1s |

**Observações sobre os hiperparâmetros ótimos:**

- **XGBoost manteve `max_depth=3`**: árvores rasas. Isso é comum em dados tabulares com features correlacionadas — profundidade excessiva gera overfitting.
- **LightGBM também ficou com `max_depth=3`** e `num_leaves=7`, um modelo raso e estável.
- **LightGBM `subsample=0.862`**: usa boa parte dos dados, mas ainda com amostragem para reduzir variância.
- **RF `max_features=log2`**: usa cerca de 3-4 features por nó, forçando diversidade entre árvores.
- **Ridge `alpha=0.01`**: regularização quase nula — os coeficientes lineares têm alta liberdade. Coerente com a força do sinal RAPM.

---

## Resultados finais e decisão dos finalistas

**Métricas após tuning** (folds 2023, 2024, 2025 — do `relatorio_modelos_tunados_26_28_05.txt`):

| Modelo | Score composto | MAE médio | DP MAE | RMSE médio | R² médio | Kendall τ | Top-3 acc | Tempo tuning |
|---|---|---|---|---|---|---|---|---|
| **LightGBM** | **0.4971** | 2.3264 | 0.1316 | 3.0146 | 0.6582 | **0.6530** | 0.2563 | 0.37 min |
| **XGBoost** | 0.4963 | 2.3479 | 0.1358 | 3.0207 | 0.6569 | 0.6523 | 0.2563 | 0.61 min |
| Random Forest | 0.4957 | 2.3732 | **0.1235** | 3.0515 | 0.6501 | 0.6436 | **0.2689** | 1.14 min |
| Ridge | 0.4900 | **2.2723** | 0.1306 | **2.9574** | **0.6710** | **0.6543** | 0.1856 | 0.07 min |

---

## Decisão dos finalistas (30/05/2026)

**Finalistas: LightGBM + XGBoost**
**Arquivado como modelo de árvore finalista: Random Forest**
**Mantido como baseline: Ridge**

### Por que LightGBM supera XGBoost?

LightGBM e XGBoost ficaram próximos no score composto. LightGBM vence por margem pequena e mantém vantagem em MAE, RMSE, R², Kendall τ e menor tempo de tuning:

| Critério | LightGBM | XGBoost | Diferença |
|---|---|---|---|
| Score composto | **0.4971** | 0.4963 | +0.0008 |
| MAE médio | **2.3264** | 2.3479 | -0.022 |
| Kendall τ | **0.6530** | 0.6523 | +0.0007 |
| Top-3 accuracy | **0.2563** | **0.2563** | 0.000 |
| Tempo tuning | **0.37 min** | 0.61 min | 1.7× mais rápido |

A diferença é pequena, então a decisão deve ser apresentada como empate técnico com vantagem operacional do LightGBM.

### Por que XGBoost volta como segundo finalista?

Após a busca fina do time-decay e retuning com `decay=0.99`, XGBoost superou Random Forest no score composto (`0.4963` vs. `0.4957`) e empatou com LightGBM em top-3 médio. Random Forest teve o maior top-3 médio (`0.2689`), mas piorou MAE, RMSE, R² e Kendall τ. Pelo critério final de cinco métricas, XGBoost volta como segundo finalista.

### Por que Ridge continua baseline apesar de melhor MAE/R²?

O Ridge tem o melhor MAE global (2.2723), melhor RMSE, melhor R² e melhor Kendall τ. Este resultado é metodologicamente honesto e deve ser reportado. O Ridge permanece como **baseline**, não como finalista principal, por três razões:

1. **Objetivo científico do TCC**: o projeto compara modelos de árvore interpretáveis (feature importance, SHAP) com baseline linear. Usar Ridge como finalista principal esvazia essa comparação — seria como concluir "o simples é suficiente" sem explorar o que os árvores capturam adicionalmente.

2. **Fase 2 — TrAdaBoost**: o algoritmo de adaptação regulatória repondera instâncias de treino. Isso funciona naturalmente com modelos de árvore (XGBoost, LightGBM, RF). Aplicar TrAdaBoost sobre Ridge seria tecnicamente possível mas metodologicamente diferente — e sem os estudos de drift/feature importance que fundamentam a narrativa da Fase 2.

3. **Interpretabilidade diferenciada**: árvores produzem feature importance nativa (gain, split count). Ridge produz coeficientes lineares — mais simples, mas não comparáveis com o que os papeis de referência ([2], [3]) usam para análise de feature importance em F1.

A força do Ridge é um resultado *a reportar*, não um problema. Na verdade, confirma a hipótese de Henderson et al. [9] e Snoeks [10]: os coeficientes lineares de construtor e piloto (RAPM) capturam a maior parte do sinal preditivo na F1. Os modelos de árvore existem para capturar as interações não-lineares residuais.

---

## Avaliação crítica

**Por que o Optuna só usou 50 trials?**

O cronograma (seção "Pontos de Atenção") menciona a opção de reduzir para 30 trials no LightGBM se necessário. 50 trials foi mantido para todos. Com TPE e `seed=42`, 50 trials é suficiente para espaços de 7-9 dimensões — estudos empíricos com TPE mostram convergência razoável já com 30-40 trials em espaços similares (Akiba et al. [23]).

**Ridge `alpha=0.01` — regularização quase nula:**

O Ridge usa grid-search temporal nos mesmos folds 2023-2024 dos demais modelos. O alpha ótimo permaneceu `0.01`, indicando regularização quase nula e força do sinal linear RAPM.

**Diferença LightGBM vs. XGBoost é mínima:**

A diferença de score composto entre os finalistas é `0.0008`. Em termos práticos, é empate técnico. A preferência por LightGBM se apoia tanto no melhor equilíbrio multi-métrica quanto no custo computacional menor.

---

## Convergência com a literatura

| Aspecto | Alinhado | Divergente | Observação |
|---|---|---|---|
| XGBoost como algoritmo finalista | ✅ | — | Previsto na arquitetura e voltou ao top-2 após busca fina do time-decay |
| Random Forest como segundo finalista | — | ⚠️ | Maior top-3 médio, mas ficou atrás de XGBoost no score composto |
| Optuna 50 trials | ✅ | — | Arquitetura seção 9 |
| Folds de tuning isolados do fold 2025 | ✅ | — | Fold 2025 nunca visto durante tuning; usado apenas na seleção temporal multi-fold |
| Ridge como baseline (não finalista) | ✅ | — | Arquitetura: "Adaptado do RAPM paper. Serve como referência simples." |
| Ridge supera árvores em MAE | ⚠️ | — | Não previsto, mas coerente com a força linear do sinal RAPM (Snoeks [10]) |
| LightGBM adicionado ao cronograma | ⚠️ | — | Não constava na arquitetura original; adicionado com base em Barra et al. [3] |
