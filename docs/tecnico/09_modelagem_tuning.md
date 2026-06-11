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

**Objetivo revisado:** após a revisão de seleção de features e a retirada da métrica top-3 do pipeline principal, o tuning passou a maximizar um score composto multi-métrica nos folds 2023-2024. O score combina MAE invertido (0.35), RMSE invertido (0.20), R² (0.20) e Kendall τ (0.25). Isso evita escolher hiperparâmetros bons apenas em MAE, mas ruins para erro quadrático, explicabilidade ou ranking.

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
| **XGBoost** | `n_estimators=100, max_depth=3, lr=0.040, subsample=0.707, colsample_bytree=0.626, reg_alpha=0.090, reg_lambda=0.612` | 251.1s |
| **LightGBM** | `n_estimators=200, max_depth=6, num_leaves=16, lr=0.022, subsample=0.710, colsample_bytree=0.896, reg_alpha=0.665, reg_lambda=0.115, min_child_samples=23` | 232.6s |
| **Random Forest** | `n_estimators=140, max_depth=6, max_features=0.5, min_samples_split=8, min_samples_leaf=5` | 240.7s |
| **Ridge** | `alpha=0.01` com `StandardScaler` | 7.9s |

**Observações sobre os hiperparâmetros ótimos:**

- **XGBoost manteve `max_depth=3`**: árvores rasas. Isso é comum em dados tabulares com features correlacionadas — profundidade excessiva gera overfitting.
- **LightGBM ficou com `max_depth=6`, mas `num_leaves=16`**: a profundidade máxima permite interações, enquanto o limite de folhas mantém regularização.
- **LightGBM `subsample=0.710`**: usa amostragem relevante para reduzir variância.
- **RF `max_features=0.5`**: usa metade das features por divisão, equilibrando diversidade entre árvores e acesso ao sinal dominante de `qualifying_position`.
- **Ridge `alpha=0.01`**: regularização quase nula — os coeficientes lineares têm alta liberdade. Coerente com a força do sinal RAPM.

---

## Resultados finais e decisão dos finalistas

**Métricas após tuning** (folds 2023, 2024, 2025 — do `relatorio_modelos_tunados_26_28_05.txt`):

| Modelo | Score composto | MAE médio | DP MAE | RMSE médio | R² médio | Kendall τ | Tempo tuning |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ridge | **0.5314** | **2.2723** | 0.1306 | **2.9574** | **0.6710** | **0.6543** | 0.13 min |
| **LightGBM** | **0.5279** | 2.3172 | 0.1233 | 3.0121 | 0.6587 | 0.6536 | 3.88 min |
| **Random Forest** | 0.5272 | 2.3263 | **0.1220** | 3.0121 | 0.6589 | 0.6503 | 4.01 min |
| XGBoost | 0.5269 | 2.3415 | 0.1448 | 3.0161 | 0.6578 | 0.6525 | 4.19 min |

---

## Decisão dos finalistas (30/05/2026)

**Finalistas de árvore: LightGBM + Random Forest**
**Arquivado como terceiro modelo de árvore: XGBoost**
**Mantido como baseline: Ridge**

### Por que LightGBM permanece como principal modelo de árvore?

LightGBM ficou com o maior score composto entre os modelos de árvore e apresentou o melhor MAE médio das árvores. A margem sobre Random Forest é pequena, mas consistente com o critério revisado:

| Critério | LightGBM | Random Forest | Diferença |
|---|---|---|---|
| Score composto | **0.5279** | 0.5272 | +0.0007 |
| MAE médio | **2.3172** | 2.3263 | -0.009 |
| RMSE médio | **3.0121** | 3.0121 | empate prático |
| Kendall τ | **0.6536** | 0.6503 | +0.0034 |
| Tempo tuning | **3.88 min** | 4.01 min | levemente menor |

A diferença é pequena, então a decisão deve ser apresentada como empate técnico com vantagem operacional do LightGBM.

### Por que Random Forest entra como segundo finalista?

Após a remoção de top-3 do score, Random Forest superou XGBoost no score composto final (`0.5272` vs. `0.5269`) e ficou praticamente empatado com LightGBM em RMSE e R². A diferença é pequena, mas Random Forest passa a ser mais defensável como segundo modelo de árvore porque oferece um comportamento de ensemble por bagging, teoricamente útil para análise de estabilidade e drift.

XGBoost permanece documentado como terceiro modelo de árvore avaliado. Ele ainda é metodologicamente relevante, mas deixou de ser finalista quando a seleção passou a considerar apenas métricas comparáveis à literatura de regressão.

### Por que Ridge continua baseline apesar de melhor MAE/R²?

O Ridge tem o melhor score composto global (0.5314), menor MAE, menor RMSE, maior R² e maior Kendall τ. Este resultado é metodologicamente honesto e deve ser reportado. O Ridge permanece como **baseline**, não como finalista principal, por três razões:

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

**Diferença entre as árvores é mínima:**

A diferença de score composto entre LightGBM, Random Forest e XGBoost é inferior a `0.0011`. Em termos práticos, é empate técnico. A preferência por LightGBM e Random Forest se apoia no critério revisado sem top-3: melhor equilíbrio médio para LightGBM e ligeira vantagem de Random Forest sobre XGBoost no score final.

---

## Convergência com a literatura

| Aspecto | Alinhado | Divergente | Observação |
|---|---|---|---|
| XGBoost como algoritmo finalista | — | ⚠️ | Previsto na arquitetura, mas ficou em terceiro entre as árvores após remover top-3 do score |
| Random Forest como segundo finalista | ✅ | — | Superou XGBoost no score revisado e oferece comparação bagging vs. boosting |
| Optuna 50 trials | ✅ | — | Arquitetura seção 9 |
| Folds de tuning isolados do fold 2025 | ✅ | — | Fold 2025 nunca visto durante tuning; usado apenas na seleção temporal multi-fold |
| Ridge como baseline (não finalista) | ✅ | — | Arquitetura: "Adaptado do RAPM paper. Serve como referência simples." |
| Ridge supera árvores em MAE | ⚠️ | — | Não previsto, mas coerente com a força linear do sinal RAPM (Snoeks [10]) |
| LightGBM adicionado ao cronograma | ⚠️ | — | Não constava na arquitetura original; adicionado com base em Barra et al. [3] |
