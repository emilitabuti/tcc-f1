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
| Validação temporal e anti-leakage | Scikit-learn: `TimeSeriesSplit` e guia de *common pitfalls*; Cawley & Talbot (2010) sobre viés de seleção em model selection |
| Ridge e normalização | Scikit-learn: documentação de `Ridge` e `StandardScaler`/padronização |

**Referências metodológicas complementares usadas nesta revisão:**

- Scikit-learn, validação cruzada em séries temporais: https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split
- Scikit-learn, vazamento de dados e pré-processamento: https://scikit-learn.org/stable/common_pitfalls.html#data-leakage
- Scikit-learn, Ridge Regression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
- Scikit-learn, padronização de features: https://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling
- Optuna `TPESampler`: https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html
- Akiba et al. (2019), Optuna: https://dl.acm.org/doi/10.1145/3292500.3330701
- Bergstra & Bengio (2012), Random Search for Hyper-Parameter Optimization: https://jmlr.org/papers/v13/bergstra12a.html
- Cawley & Talbot (2010), seleção de modelos e viés de avaliação: https://jmlr.org/papers/v11/cawley10a.html

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

O fold 2025 é reservado como holdout também para o tuning — os hiperparâmetros são escolhidos sem qualquer acesso ao desempenho em 2025. Isso é uma cadeia de isolamento completa: decay escolhido em 2023-2024 → hiperparâmetros escolhidos em 2023-2024 → avaliação final em 2023, 2024, 2025.

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

**Ridge** (grid-search manual, não Optuna):

O baseline linear não usa Optuna porque só há um hiperparâmetro principal a escolher: `alpha`, a força da penalização L2. O script final `src/otimizacao_ridge_lambda.py` varre 25 valores em escala logarítmica entre 0.01 e 100 (`np.logspace(-2, 2, 25)`) e avalia cada valor nos mesmos folds temporais de tuning dos modelos de árvore:

```
2018-2022 -> validação 2023
2018-2023 -> validação 2024
```

Resultado: `alpha=0.01`. Isso indica regularização mínima. Em termos simples, o Ridge quase não precisou "segurar" os coeficientes lineares: as features finais, especialmente os sinais de piloto/construtor derivados do RAPM e a informação de classificação, já formam uma relação linear forte com `finish_position`.

**Importante:** a implementação final não usa `RidgeCV`. Ela usa `Ridge(alpha=...)` dentro de um wrapper com `StandardScaler`, ajustando o scaler somente no treino de cada fold e aplicando `transform` na validação. Isso segue a recomendação do scikit-learn para evitar vazamento em pré-processamento: aprender médias/desvios apenas com o conjunto de treino e nunca com o conjunto de validação/teste.

---

## Hiperparâmetros ótimos encontrados

| Modelo | Parâmetros ótimos | Tempo de tuning |
|---|---|---|
| **XGBoost** | `n_estimators=269, max_depth=3, lr=0.022, subsample=0.632, colsample_bytree=0.654, reg_alpha=0.722, reg_lambda=0.578` | 57.2s |
| **LightGBM** | `n_estimators=146, max_depth=4, num_leaves=8, lr=0.032, subsample=0.993, colsample_bytree=0.863, reg_alpha=0.806, reg_lambda=0.559, min_child_samples=32` | 21.5s |
| **Random Forest** | `n_estimators=173, max_depth=7, max_features=0.5, min_samples_split=3, min_samples_leaf=5` | 89.7s |
| **Ridge** | `alpha=0.01` com `StandardScaler` | 3.9s |

**Observações sobre os hiperparâmetros ótimos:**

- **XGBoost e LightGBM têm `max_depth=3` e `4`**: árvores rasas. Isso é comum em dados tabulares com features correlacionadas — profundidade excessiva gera overfitting.
- **LightGBM `num_leaves=8`**: com `max_depth=4`, poderia ter até 15 folhas. Apenas 8 — modelo simples.
- **LightGBM `subsample=0.993`**: essencialmente usando todos os dados. O GOSS interno já faz a amostragem seletiva.
- **RF `max_features=0.5`**: usa 50% das features em cada nó (7-8 de 15). Mais features por nó do que `sqrt` (≈3.9) — RF achou útil ver mais do espaço de features.
- **Ridge `alpha=0.01`**: regularização quase nula — os coeficientes lineares têm alta liberdade. Coerente com a força do sinal RAPM.

---

## Validação dos hiperparâmetros ótimos

Para conferir se os hiperparâmetros documentados realmente correspondem aos melhores resultados salvos nos artefatos, foi criada uma verificação isolada em `src/verificar/validar_09_modelagem_tuning.py`.

1. os melhores trials de Optuna contra `optuna_*_best_params.json`;
2. o grid de `alpha` do Ridge contra `ridge_best_params.json`;
3. a presença dos folds temporais 2023-2024 em `tuning_utils.py`;
4. se o script do Ridge usa ou não `RidgeCV`.

Resumo da validação:

| Modelo | Busca | Melhor trial/grid | MAE médio nos folds de tuning | Parâmetros conferem com JSON? |
|---|---|---:|---:|---|
| XGBoost | Optuna TPE, 50 trials | trial 23 | 2.306488 | Sim |
| LightGBM | Optuna TPE, 50 trials | trial 34 | 2.291017 | Sim |
| Random Forest | Optuna TPE, 50 trials | trial 49 | 2.283507 | Sim |
| Ridge | Grid log de 25 alphas | `alpha=0.01` | 2.210738 | Sim |

Validações específicas do Ridge:

| Checagem | Resultado |
|---|---|
| `alpha` salvo em `ridge_best_params.json` | 0.01 |
| `alpha` vencedor em `ridge_alpha_grid.csv` | 0.01 |
| Quantidade de alphas avaliados | 25 |
| Script usa `RidgeCV`? | Não |
| Script usa `FOLDS_TUNING`? | Sim |

Sensibilidade do Ridge ao `alpha` nos folds de tuning:

| Alpha | MAE médio tuning | Δ MAE vs. `alpha=0.01` |
|---:|---:|---:|
| 0.01 | 2.210738 | 0.000000 |
| 0.10 | 2.210750 | +0.000013 |
| 1.00 | 2.210876 | +0.000138 |
| 10.00 | 2.212207 | +0.001469 |
| 100.00 | 2.229301 | +0.018563 |

Leitura: `alpha=0.01` é o vencedor formal, mas a curva é praticamente plana entre 0.01 e 10.0. Isso torna o baseline linear robusto à escolha exata de `alpha`: mesmo uma regularização muito maior (`alpha=10.0`) piora o MAE de tuning em apenas 0.0015 posição. A queda mais visível aparece só em `alpha=100.0`, quando a penalização já começa a encolher demais os coeficientes.

### Por que foi feito dessa maneira?

O desenho segue uma lógica de previsão temporal, não de amostragem aleatória. Em F1, uma corrida futura nunca pode informar a escolha de hiperparâmetros de uma corrida passada. Por isso, os folds são expansivos: treina-se até 2022 e valida-se 2023; depois treina-se até 2023 e valida-se 2024. O ano de 2025 fica fora da seleção de hiperparâmetros e entra apenas na avaliação final.

Essa separação reduz dois riscos clássicos:

- **vazamento temporal:** usar informação de 2025, ou estatísticas calculadas com 2025, para decidir hiperparâmetros;
- **viés de seleção:** ajustar hiperparâmetros diretamente ao conjunto que será usado para reportar a performance final.

A literatura de validação temporal do scikit-learn recomenda exatamente essa ideia: em dados autocorrelacionados no tempo, métodos clássicos como `KFold`/`ShuffleSplit` podem gerar estimativas ruins porque misturam observações próximas no tempo entre treino e validação. Cawley & Talbot (2010) também alertam que a própria etapa de seleção de modelo pode sofrer overfitting quando o mesmo critério é usado de forma pouco isolada.

---
## Resultados finais e decisão dos finalistas

**Métricas após tuning** (folds 2023, 2024, 2025 — do `relatorio_modelos_tunados_26_28_05.txt`):

| Modelo | MAE médio | DP MAE | RMSE médio | R² médio | Kendall τ | Top-3 acc | Tempo tuning |
|---|---|---|---|---|---|---|---|
| Ridge | **2.2734** | 0.1336 | **2.9582** | **0.6708** | 0.6546 | 0.1856 | 0.06 min |
| **LightGBM** | **2.3133** | **0.1117** | 3.0082 | 0.6598 | **0.6551** | **0.2424** | 0.36 min |
| **Random Forest** | **2.3275** | 0.1210 | 3.0196 | 0.6573 | 0.6511 | 0.2134 | 1.50 min |
| XGBoost | 2.3342 | 0.1334 | 3.0137 | 0.6584 | 0.6518 | 0.2412 | 0.95 min |

---

## Decisão dos finalistas (30/05/2026)

**Finalistas: LightGBM + Random Forest**
**Arquivado: XGBoost**
**Mantido como baseline: Ridge**

### Por que LightGBM supera XGBoost?

Em todos os critérios definidos (MAE médio, Kendall τ, top-3, tempo de tuning), LightGBM supera XGBoost:

| Critério | LightGBM | XGBoost | Diferença |
|---|---|---|---|
| MAE médio | **2.3133** | 2.3342 | -0.021 |
| Kendall τ | **0.6551** | 0.6518 | +0.003 |
| Top-3 accuracy | **0.2424** | 0.2412 | +0.001 |
| DP MAE (estabilidade) | **0.1117** | 0.1334 | -0.022 |
| Tempo tuning | **0.36 min** | 0.95 min | 2.6× mais rápido |

A diferença de MAE (0.021 posições) é pequena mas consistente em todos os folds. O argumento mais forte para LightGBM na Fase 2 é o tempo de tuning: TrAdaBoost iterativo re-treina o modelo a cada corrida nova de 2026 — 0.36 min por ciclo vs. 0.95 min tem impacto cumulativo.

### Por que Random Forest é o segundo finalista e não o Ridge?

O Ridge tem o melhor MAE global (2.2734). Este resultado é metodologicamente honesto e deve ser reportado. O Ridge permanece como **baseline**, não como finalista principal, por três razões:

1. **Objetivo científico do TCC**: o projeto compara modelos de árvore interpretáveis (feature importance, SHAP) com baseline linear. Usar Ridge como finalista principal esvazia essa comparação — seria como concluir "o simples é suficiente" sem explorar o que os árvores capturam adicionalmente.

2. **Fase 2 — TrAdaBoost**: o algoritmo de adaptação regulatória repondera instâncias de treino. Isso funciona naturalmente com modelos de árvore (XGBoost, LightGBM, RF). Aplicar TrAdaBoost sobre Ridge seria tecnicamente possível mas metodologicamente diferente — e sem os estudos de drift/feature importance que fundamentam a narrativa da Fase 2.

3. **Interpretabilidade diferenciada**: árvores produzem feature importance nativa (gain, split count). Ridge produz coeficientes lineares — mais simples, mas não comparáveis com o que os papeis de referência ([2], [3]) usam para análise de feature importance em F1.

A força do Ridge é um resultado *a reportar*, não um problema. Na verdade, confirma a hipótese de Henderson et al. [9] e Snoeks [10]: os coeficientes lineares de construtor e piloto (RAPM) capturam a maior parte do sinal preditivo na F1. Os modelos de árvore existem para capturar as interações não-lineares residuais.

---

## Avaliação crítica

**Diferença LightGBM vs. XGBoost de 0.021 posições:**

A diferença de MAE entre os finalistas é menor que a precisão prática de qualquer previsão de F1 (os sistemas de apostas tipicamente têm margem de ~2 posições). Argumentar que LightGBM "supera" XGBoost com base nessa diferença requer qualificação — a diferença é consistente e estatisticamente observada, mas marginalmente pequena em termos práticos.

---

## Convergência com a literatura

| Aspecto | Alinhado | Divergente | Observação |
|---|---|---|---|
| XGBoost como algoritmo principal | — | ⚠️ | Substituído por LightGBM empiricamente; XGBoost era o previsto na arquitetura |
| Random Forest como segundo finalista | ✅ | — | Conforme arquitetura |
| Optuna 50 trials | ✅ | — | Arquitetura seção 9 |
| Folds de tuning isolados do fold de avaliação final | ✅ | — | Fold 2025 nunca visto durante tuning |
| Ridge como baseline (não finalista) | ✅ | — | Arquitetura: "Adaptado do RAPM paper. Serve como referência simples." |
| Ridge supera árvores em MAE | ⚠️ | — | Não previsto, mas coerente com a força linear do sinal RAPM (Snoeks [10]) |
| LightGBM adicionado ao cronograma | ⚠️ | — | Não constava na arquitetura original; adicionado com base em Barra et al. [3] |
