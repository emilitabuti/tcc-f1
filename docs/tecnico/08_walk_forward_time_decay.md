# 08 — Walk-Forward e Time-Decay

## Contexto

Com o dataset de modelagem pronto (15 features, 2.943 linhas), o próximo passo é definir como treinar e avaliar os modelos respeitando a natureza temporal dos dados. Dois problemas precisam ser resolvidos: (1) como dividir treino e validação sem contaminar o futuro no passado; (2) como ponderar observações mais antigas para que o modelo priorize padrões recentes.

---

## Fundamentação bibliográfica

| Decisão | Referência |
|---|---|
| Walk-forward validation como requisito em séries temporais esportivas | Henderson et al. [9] — único paper da revisão que formaliza walk-forward para F1 |
| Time-decay com fator 0.75 por temporada como ponto de partida | Henderson et al. [9] — "valor ótimo encontrado no RAPM paper" |
| Pesos decrescentes a dados mais antigos — fundamentação matemática | Tan et al. [18] — Instance-Conditional Timescales of Decay for Non-Stationary Learning |
| Proibição de embaralhar dados temporais | Arquitetura, seção 8: "Nunca embaralhar os dados. A ordem temporal é inviolável." |

---

## Por que walk-forward e não cross-validation padrão?

Cross-validation padrão com k-folds embaralha o dataset antes de dividir. Em séries temporais isso cria **leakage temporal**: o modelo pode ser treinado em dados de 2024 e validado em dados de 2022, ou seja, "vê o futuro" durante o treinamento.

Num modelo de F1, as consequências são concretas:
- Coeficientes RAPM calculados causalmente seriam contaminados se o modelo aprendesse resultados futuros
- Features como `recent_form_5` dependem da ordem temporal — embaralhar rompe o significado da feature

Walk-forward preserva a ordem temporal estritamente:

```
Fold 1: Treino 2018-2022 → Validação 2023
Fold 2: Treino 2018-2023 → Validação 2024
Fold 3: Treino 2018-2024 → Validação 2025  ← baseline final
```

Cada fold *acrescenta* dados ao treino — nunca os mistura com o conjunto de validação.

---

## Definição dos folds

**Por que três folds e não quatro (como previsto na arquitetura)?**

A arquitetura (seção 8) previa quatro folds com início em 2014. A implementação adaptou para 2018 (razão documentada no documento 01 — FastF1 não tem cobertura confiável antes de 2018). Com dados de 2018, três folds cobrem os anos de validação relevantes (2023, 2024, 2025). O quarto fold (treino 2018-2025 → drift test 2026) está planejado para a Semana 3 (P2) com os dados disponíveis via OpenF1.

**Tamanho dos folds** (do `relatorio_segunda_semana2_xgboost.txt`):

| Fold | Treino | Validação | N treino | N validação |
|---|---|---|---|---|
| 1 | 2018-2022 | 2023 | 1.725 | 374 |
| 2 | 2018-2023 | 2024 | 2.099 | 425 |
| 3 | 2018-2024 | 2025 | 2.524 | 419 |

---

## Time-decay — como funciona

O time-decay pondera observações de treino: corridas mais recentes recebem peso maior que corridas mais antigas.

**Implementação** (função `calcular_sample_weight` em `walk_forward.py` e `otimizacao_time_decay.py`):

```python
def calcular_sample_weight(y_train, valid_season, decay):
    distancia = valid_season - y_train["season"]  # diferença em temporadas
    distancia = distancia.clip(lower=0)            # garante não-negativo
    return np.power(decay, distancia).to_numpy()
```

Para o fold 3 (treino 2018-2024, validação 2025) com decay=0.95:

| Temporada | Distância até 2025 | Peso (decay=0.95) |
|---|---|---|
| 2024 | 1 | 0.9500 |
| 2023 | 2 | 0.9025 |
| 2022 | 3 | 0.8574 |
| 2021 | 4 | 0.8145 |
| 2020 | 5 | 0.7738 |
| 2019 | 6 | 0.7351 |
| 2018 | 7 | 0.6983 |

O peso é passado ao `model.fit()` via `sample_weight` — o modelo otimiza a função de perda ponderada. Temporadas antigas não são descartadas: a de 2018 ainda recebe peso de ~0.70. Isso é muito diferente de decay=0.75 (onde 2018 valeria ~0.13).

**Importante:** o time-decay no walk-forward usa `valid_season` como referência, não a temporada atual. No fold 1, a temporada de 2022 tem distância 1 para a validação de 2023; no fold 3, distância 3.

---

## Otimização do time-decay

**Script:** `src/otimizacao_time_decay.py`

**Candidatos testados:** {0.50, 0.65, 0.75, 0.85, 0.95}

**Folds de otimização:** apenas 2023 e 2024 (fold 1 e fold 2). O fold 2025 foi **reservado como holdout** — não participou da seleção do decay para evitar data leakage na otimização do hiperparâmetro.

**Resultados** (do `otimizacao_time_decay_xgboost_resumo.csv`):

| Decay | MAE médio 2023-2024 | DP MAE | RMSE médio | Kendall τ |
|---|---|---|---|---|
| **0.95** | **2.4026** | 0.2451 | **3.0627** | **0.6471** |
| 0.85 | 2.4188 | 0.1986 | 3.1016 | 0.6420 |
| 0.75 | 2.4235 | **0.1248** | 3.0997 | 0.6419 |
| 0.65 | 2.4339 | 0.1515 | 3.1141 | 0.6386 |
| 0.50 | 2.4652 | 0.1659 | 3.1538 | 0.6324 |

**Vencedor: decay=0.95** — menor MAE médio em 2023-2024.

**Atenção ao DP:** o decay=0.95 tem o maior desvio padrão (0.245) entre os testados. O decay=0.75 tem o menor DP (0.125). Isso significa que 0.95 é ótimo em média mas menos estável entre folds. A banca pode questionar: "você escolheu o decay mais instável". Resposta: o critério de seleção definido na arquitetura (seção 8) era minimizar MAE médio, não minimizar instabilidade. O valor 0.95 atende ao critério especificado.

---

## Por que 0.95 e não 0.75 do paper?

Henderson et al. [9] identificaram 0.75 como valor ótimo no RAPM paper. A divergência tem três explicações plausíveis:

**1. Dataset diferente (2018+ vs. 2014+):**
O período 2018-2025 está integralmente dentro da era híbrida — regulamento, filosofia de design e métricas competitivas são mais estáveis que no período 2014-2024. Corridas de 2018 ainda são relevantes em 2025. Com decay=0.75, a temporada de 2018 valeria apenas 0.13 no fold 3 — essencialmente descartada. Com 0.95, vale 0.70 — ainda informativamente útil.

**2. Tamanho menor do dataset:**
Com apenas 7 temporadas de dados (vs. 11 de Henderson et al.), descartar agressivamente os anos iniciais deixaria o treino muito pequeno para o modelo generalizar.

**3. Features de decaimento interno:**
O projeto já inclui `season_factor` e `driver_coef_rapm` que capturam evolução temporal. O time-decay no walk-forward pode precisar ser menos agressivo porque as features já comunicam o efeito temporal.

**O argumento central para a defesa:**
O decay foi otimizado empiricamente nos dados deste projeto, não apenas copiado do paper. O resultado empírico é 0.95. A literatura cita 0.75 como ponto de partida, não como valor universal — o próprio Henderson et al. [9] recomendam otimizar o fator para o dataset específico via grid-search temporal, que foi exatamente o que foi feito.

---

## Por que os dados nunca são embaralhados?

A proibição de embaralhar é inviolável em séries temporais preditivas. Três razões:

1. **Features causais dependem de ordem:** `recent_form_5` de um piloto em 2025 usa seus 5 resultados mais recentes. Se embaralharmos, uma corrida de 2024 pode aparecer "depois" de uma de 2025 no treino, tornando a feature inválida.

2. **Coeficientes RAPM:** calculados corrida a corrida com dados anteriores. Embaralhar a base de avaliação não corrompe o RAPM (que foi pré-calculado), mas corromperia qualquer re-cálculo incremental.

3. **Leakage de hiperparâmetros:** se o modelo "ver" 2024 durante o treino do fold 1 (que deveria validar em 2023), toda a curva de degradação para 2026 seria inválida.

---

## Definição das métricas de avaliação

**Script:** `src/metricas.py`

Todas as métricas são calculadas sobre o conjunto de validação de cada fold (nunca sobre o treino).

### MAE — Mean Absolute Error

```python
mae = mean_absolute_error(y_true, y_pred)  # posições
```

Interpretação direta: em média, o modelo erra X posições. Meta: ≤ 2.5 posições (alinhada com Henderson et al. [9]: MAE=2.3; TabNet [7]: MAE=2.17).

### RMSE — Root Mean Squared Error

Penaliza erros grandes mais que o MAE. Meta: ≤ 3.0.

### R² — Coeficiente de Determinação

Proporção da variância de `finish_position` explicada pelo modelo. Meta: ≥ 0.75 (baseada no TabNet [7]).

### Kendall τ — Correlação de Rankings

```python
def kendall_tau_por_corrida(df_pred):
    valores = []
    for _, grupo in df_pred.groupby(["season", "round"]):
        tau, _ = kendalltau(grupo["finish_position"], grupo["pred_finish_position"])
        valores.append(tau)
    return np.mean(valores)
```

**Calculado por corrida individualmente** — para cada GP, mede a concordância entre o ranking real e o ranking predito. O resultado final é a **média de todos os GPs** do fold. Isso é diferente de calcular Kendall τ globalmente sobre todas as corridas juntas (que inflaria artificialmente o valor por misturar corridas de diferentes GPs).

Meta: ≥ 0.60 (baseada em Henderson et al. [9]: τ=0.625).

### Acurácia Top-3

```python
def acuracia_top3(df_pred):
    acertos = []
    for _, grupo in df_pred.groupby(["season", "round"]):
        real_top3 = set(grupo.nsmallest(3, "finish_position")["driver_id"])
        pred_top3 = set(grupo.nsmallest(3, "pred_finish_position")["driver_id"])
        acertos.append(int(real_top3 == pred_top3))  # ← igualdade exata de conjunto
    return np.mean(acertos)
```

**Igualdade exata de conjunto**: os três pilotos preditos como pódio precisam ser exatamente os três que fizeram pódio, sem importar a ordem interna. Uma predição que acerta 2 dos 3 não pontua.

**Por que os valores são baixos (18-24%)?** Isso é esperado para um problema de regressão contínua avaliado por classificação de pódio exata. Polishchuk [1] que reporta 78% usou um modelo de classificação direta treinado especificamente para prever pódio — o problema de otimização era diferente. Comparações diretas com esse benchmark são metodologicamente inválidas sem ajuste.

---

## Resultados obtidos

### Otimização de time-decay

| Decay | MAE médio (folds 2023-2024) | Escolhido |
|---|---|---|
| 0.95 | 2.4026 ± 0.245 | ✅ Sim |
| 0.85 | 2.4188 ± 0.199 | Não |
| 0.75 | 2.4235 ± 0.125 | Não |

### Walk-forward por fold (modelo XGBoost baseline, antes do tuning)

Do `relatorio_segunda_semana2_xgboost.txt`:

| Fold | MAE | RMSE | R² | Kendall τ | N validação |
|---|---|---|---|---|---|
| Treino 2018-2022 → 2023 | ~2.43 | ~3.07 | ~0.64 | ~0.64 | 374 |
| Treino 2018-2023 → 2024 | ~2.18 | ~2.87 | ~0.70 | ~0.68 | 425 |
| Treino 2018-2024 → 2025 | ~2.39 | ~3.10 | ~0.64 | ~0.63 | 419 |

Os valores definitivos após tuning estão no documento 09.

---

## Avaliação crítica

**O fold 2025 não participou da otimização do decay:**

O decay foi otimizado em folds 2023-2024. O fold 2025 é genuinamente independente — o valor 0.95 não foi ajustado para otimizar 2025. Isso é metodologicamente correto e defensável.

**A escolha de 0.95 tem maior instabilidade:**

DP=0.245 vs. 0.125 do decay=0.75. Isso significa que o modelo com 0.95 tem performance mais variável entre folds — pode ser excelente em 2024 mas mediano em 2023 (ou vice-versa). Para a Fase 2 com TrAdaBoost, essa instabilidade pode ser relevante: o algoritmo precisa de um modelo base estável.

**Apenas 5 valores testados:**

O grid-search cobriu 5 pontos em [0.50, 0.95]. Espaços intermediários como 0.88 ou 0.92 não foram testados. Uma busca mais fina poderia revelar valor ligeiramente diferente. Para o TCC isso é aceitável — o esforço computacional de 50 folds extras para afinar o decimal não justifica a melhoria marginal esperada.

---

## Convergência com a literatura

| Aspecto | Alinhado | Divergente | Observação |
|---|---|---|---|
| Walk-forward obrigatório em séries temporais | ✅ | — | Henderson et al. [9], arquitetura seção 8 |
| Nunca embaralhar dados | ✅ | — | Arquitetura: "A ordem temporal é inviolável" |
| Decay como `sample_weight` no `model.fit()` | ✅ | — | Henderson et al. [9], Tan et al. [18] |
| Otimizar decay via grid-search temporal | ✅ | — | Arquitetura prevê; executado nos folds 2023-2024 |
| Decay ótimo = 0.75 | — | ⚠️ | Empírico: 0.95. Defensável pela diferença de dataset e período |
| Kendall τ por corrida (não global) | ✅ | — | Henderson et al. [9]: τ calculado por GP individualmente |
| Top-3 accuracy por igualdade de conjunto | ⚠️ | — | Criterio mais estrito que o de Polishchuk [1] — explica por que 18-24% vs. 78% |
