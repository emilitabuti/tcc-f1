# 08 — Walk-Forward e Time-Decay

## Contexto

Com o dataset de modelagem pronto (13 features, 2.943 linhas), o próximo passo é definir como treinar e avaliar os modelos respeitando a natureza temporal dos dados. Dois problemas precisam ser resolvidos: (1) como dividir treino e validação sem contaminar o futuro no passado; (2) como ponderar observações mais antigas para que o modelo priorize padrões recentes.

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

Para o fold 3 (treino 2018-2024, validação 2025) com decay=0.99:

| Temporada | Distância até 2025 | Peso (decay=0.99) |
|---|---|---|
| 2024 | 1 | 0.9900 |
| 2023 | 2 | 0.9801 |
| 2022 | 3 | 0.9703 |
| 2021 | 4 | 0.9606 |
| 2020 | 5 | 0.9510 |
| 2019 | 6 | 0.9415 |
| 2018 | 7 | 0.9321 |

O peso é passado ao `model.fit()` via `sample_weight` — o modelo otimiza a função de perda ponderada. Temporadas antigas não são descartadas: a de 2018 ainda recebe peso de ~0.93. Isso é muito diferente de decay=0.75 (onde 2018 valeria ~0.13).

**Importante:** o time-decay no walk-forward usa `valid_season` como referência, não a temporada atual. No fold 1, a temporada de 2022 tem distância 1 para a validação de 2023; no fold 3, distância 3.

---

## Otimização do time-decay

**Script:** `src/otimizacao_time_decay.py`

**Candidatos testados:** {0.50, 0.65, 0.75, 0.85, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99}

**Folds de otimização:** apenas 2023 e 2024 (fold 1 e fold 2). O fold 2025 foi **reservado como holdout** — não participou da seleção do decay para evitar data leakage na otimização do hiperparâmetro.

**Critério de seleção:** score composto multi-métrica, usando os mesmos pesos adotados depois na seleção de features e no tuning dos modelos:

| Métrica normalizada | Peso |
|---|---:|
| MAE invertido | 0.35 |
| RMSE invertido | 0.20 |
| R² | 0.20 |
| Kendall τ | 0.25 |

**Resultados** (do `otimizacao_time_decay_xgboost_resumo.csv`):

| Decay | Score composto | MAE médio | DP MAE | RMSE médio | R² médio | Kendall τ |
|---|---:|---:|---:|---:|---:|---:|
| **0.99** | **1.0000** | **2.3609** | 0.1943 | **3.0328** | **0.6528** | **0.6523** |
| 0.96 | 0.8914 | 2.3663 | 0.2215 | 3.0499 | 0.6484 | 0.6495 |
| 0.95 | 0.8209 | 2.3779 | 0.2031 | 3.0598 | 0.6465 | 0.6498 |
| 0.85 | 0.7925 | 2.3896 | 0.1821 | 3.0615 | 0.6464 | 0.6515 |
| 0.65 | 0.4532 | 2.4175 | **0.1639** | 3.1058 | 0.6361 | 0.6432 |
| 0.50 | 0.0000 | 2.4544 | 0.1715 | 3.1572 | 0.6240 | 0.6298 |

**Vencedor: decay=0.99** — melhor score composto geral nos folds 2023-2024, além de melhor MAE, RMSE, R² e Kendall τ médios. A busca fina mostrou que o limite superior original (`0.95`) era bom, mas ainda havia ganho ao testar valores menos agressivos.

**Atenção ao DP:** o decay=0.99 não tem o menor DP de MAE; o decay=0.65 é mais estável. A resposta para a banca passa a ser: a escolha não foi feita por estabilidade isolada, e sim pelo melhor equilíbrio geral entre erro, explicabilidade e ranking.

---

## Por que 0.99 e não 0.75 do paper?

Henderson et al. [9] identificaram 0.75 como valor ótimo no RAPM paper. A divergência tem três explicações plausíveis:

**1. Dataset diferente (2018+ vs. 2014+):**
O período 2018-2025 está integralmente dentro da era híbrida — regulamento, filosofia de design e métricas competitivas são mais estáveis que no período 2014-2024. Corridas de 2018 ainda são relevantes em 2025. Com decay=0.75, a temporada de 2018 valeria apenas 0.13 no fold 3 — essencialmente descartada. Com 0.99, vale 0.93 — quase integralmente preservada.

**2. Tamanho menor do dataset:**
Com apenas 7 temporadas de dados (vs. 11 de Henderson et al.), descartar agressivamente os anos iniciais deixaria o treino muito pequeno para o modelo generalizar.

**3. Features de decaimento interno:**
O projeto já inclui `season_factor` e `driver_coef_rapm` que capturam evolução temporal. O time-decay no walk-forward pode precisar ser menos agressivo porque as features já comunicam o efeito temporal.

**O argumento central para a defesa:**
O decay foi otimizado empiricamente nos dados deste projeto, não apenas copiado do paper. O resultado empírico multi-métrica é 0.99. A literatura cita 0.75 como ponto de partida, não como valor universal — o próprio Henderson et al. [9] recomenda otimizar o fator para o dataset específico via validação temporal, que foi exatamente o que foi feito.

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

### Remoção da métrica top-3

A acurácia top-3 foi removida das métricas oficiais do pipeline. A decisão foi tomada porque os trabalhos bibliográficos comparáveis à regressão de `finish_position` reportam MAE, RMSE, R² e métricas de ranking/correlação. Top-3 aparece em estudos que formulam o problema como classificação de pódio ou classes de resultado, o que não é equivalente à regressão causal pré-corrida usada neste projeto.

Com isso, o score composto e a seleção de hiperparâmetros passam a avaliar apenas erro contínuo e qualidade de ranking.

---

## Resultados obtidos

### Otimização de time-decay

| Decay | Score composto | MAE médio (folds 2023-2024) | Escolhido |
|---|---:|---:|---|
| 0.99 | **1.0000** | **2.3609 ± 0.194** | ✅ Sim |
| 0.96 | 0.8355 | 2.3663 ± 0.221 | Não |
| 0.95 | 0.7747 | 2.3779 ± 0.203 | Não |
| 0.85 | 0.6712 | 2.3896 ± 0.182 | Não |

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

O decay foi otimizado em folds 2023-2024. O fold 2025 é genuinamente independente — o valor 0.99 não foi ajustado para otimizar 2025. Isso é metodologicamente correto e defensável.

**A escolha de 0.99 tem trade-off de estabilidade:**

DP=0.194, maior que os decays 0.65, 0.50 e 0.85. Isso significa que o modelo com 0.99 ainda tem alguma variação entre folds, embora não tenha sido escolhido por uma métrica isolada. O ganho de equilíbrio geral compensou esse trade-off: melhor score composto, melhor MAE, melhor RMSE, melhor R² e melhor Kendall τ.

**Apenas 5 valores testados:**

O grid-search original cobria 5 pontos em [0.50, 0.95]. A revisão posterior testou valores intermediários e menos agressivos até 0.99. Essa busca fina revelou ganho real: `0.99` superou `0.95` no score composto multi-métrica.

---

## Convergência com a literatura

| Aspecto | Alinhado | Divergente | Observação |
|---|---|---|---|
| Walk-forward obrigatório em séries temporais | ✅ | — | Henderson et al. [9], arquitetura seção 8 |
| Nunca embaralhar dados | ✅ | — | Arquitetura: "A ordem temporal é inviolável" |
| Decay como `sample_weight` no `model.fit()` | ✅ | — | Henderson et al. [9], Tan et al. [18] |
| Otimizar decay via grid-search temporal | ✅ | — | Arquitetura prevê; executado nos folds 2023-2024 |
| Decay ótimo = 0.75 | — | ⚠️ | Empírico: 0.99. Defensável pela diferença de dataset e período |
| Kendall τ por corrida (não global) | ✅ | — | Henderson et al. [9]: τ calculado por GP individualmente |
| Métricas oficiais restritas à regressão/ranking | ✅ | — | MAE, RMSE, R² e Kendall τ são comparáveis aos trabalhos de regressão revisados |
