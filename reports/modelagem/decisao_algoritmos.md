# Decisao Final dos Algoritmos - Fase 1

Data de fechamento: 30/05/2026

## Resultado

Algoritmos finalistas para apresentacao da Fase 1:

1. **LightGBM**
2. **Random Forest**

Algoritmo de arvore arquivado:

- **XGBoost**

Baseline mantido como referencia metodologica:

- **Ridge Regression**

## Evidencias Quantitativas

| Modelo | MAE medio | MAE DP | RMSE medio | R2 medio | Kendall tau | Top-3 accuracy | Tempo tuning |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ridge baseline | 2.2734 | 0.1336 | 2.9582 | 0.6708 | 0.6546 | 0.1856 | 0.06 min |
| LightGBM | 2.3133 | 0.1117 | 3.0082 | 0.6598 | 0.6551 | 0.2424 | 0.36 min |
| Random Forest | 2.3275 | 0.1210 | 3.0196 | 0.6573 | 0.6511 | 0.2134 | 1.50 min |
| XGBoost | 2.3342 | 0.1334 | 3.0137 | 0.6584 | 0.6518 | 0.2412 | 0.95 min |

Arquivos de origem:

- `reports/modelagem/tabela_metricas_tunadas_4modelos.csv`
- `reports/modelagem/tabela_metricas_tunadas_4modelos_resumo.csv`

## Justificativa da Escolha

O criterio definido no cronograma revisado foi aplicado de forma empirica:

- menor MAE medio nos folds walk-forward 2023, 2024 e 2025;
- maior Kendall tau medio;
- estabilidade entre folds, medida pelo desvio padrao do MAE;
- custo computacional;
- coerencia com a arquitetura da Fase 2, que exigira analise de drift e possivel
  adaptacao incremental.

Entre os modelos de arvore, o **LightGBM** apresentou o melhor MAE medio
(`2.3133`), o melhor Kendall tau medio (`0.6551`), o melhor top-3 accuracy medio
(`0.2424`) e o menor tempo de tuning (`0.36 min`). Por isso, ele substitui o
XGBoost como principal modelo de arvore para a Fase 1.

O **Random Forest** foi mantido como segundo finalista por combinar desempenho
proximo ao LightGBM com maior simplicidade interpretativa e robustez natural de
bagging. Ele tambem funciona como contraponto metodologico ao boosting, conforme
previsto na arquitetura.

O **XGBoost** foi arquivado como terceiro candidato. Apesar de ser o algoritmo
originalmente previsto como principal e ter desempenho adequado, ficou atras de
LightGBM e Random Forest em MAE medio e atras do LightGBM em estabilidade,
Kendall tau e custo computacional.

O **Ridge baseline** obteve o menor MAE geral (`2.2734`). Esse resultado deve ser
reportado explicitamente, pois mostra que a decomposicao linear inspirada em RAPM
e forte para este dataset. Ainda assim, ele permanece como baseline, nao como
modelo finalista principal, porque o objetivo da Fase 1 inclui comparar modelos
de arvore interpretaveis e preparar a analise de drift/adaptacao da Fase 2.

## Feature Importance

A etapa de importancia de features confirmou coerencia entre os modelos e a
literatura usada na arquitetura.

Top features por modelo:

| Modelo | Top 5 features |
|---|---|
| XGBoost | `qualifying_position`, `recent_form_5`, `constructor_coef_rapm`, `driver_constructor_synergy`, `constructor_wins_total` |
| Random Forest | `qualifying_position`, `constructor_coef_rapm`, `recent_form_5`, `driver_constructor_synergy`, `constructor_wins_total` |
| LightGBM | `qualifying_position`, `constructor_coef_rapm`, `recent_form_5`, `driver_constructor_synergy`, `constructor_wins_total` |

Arquivos gerados:

- `reports/modelagem/feature_importance_xgb.csv`
- `reports/modelagem/feature_importance_rf.csv`
- `reports/modelagem/feature_importance_lgb.csv`
- `reports/modelagem/feature_importance_2024.csv`
- `reports/modelagem/relatorio_feature_importance_29_30_05.txt`

Leitura metodologica:

- `qualifying_position` domina os tres modelos, coerente com a relevancia do grid
  e do qualifying na literatura de predicao de resultados em F1.
- `constructor_coef_rapm` aparece como feature central, coerente com a literatura
  que atribui grande peso ao construtor na variancia de desempenho.
- `recent_form_5` aparece de forma consistente, sustentando a inclusao de forma
  recente como feature historica causal.
- `driver_constructor_synergy` aparece no top 5 dos tres modelos, reforcando a
  decisao de modelar a relacao piloto-equipe.
- As taxas de DNF e features de circuito permanecem com importancia menor, mas
  preservam interpretabilidade e cobertura dos objetivos especificos do TCC.

## Dataset Final

O dataset de modelagem foi validado com:

- 2.943 linhas;
- 15 features em X;
- 0 valores NaN em X e y;
- temporadas 2018 a 2025;
- nenhuma coluna proibida em X.

Features finais:

- `qualifying_position`
- `grid_penalty`
- `recent_form_5`
- `driver_coef_rapm`
- `driver_dnf_rate`
- `constructor_coef_rapm`
- `constructor_dnf_rate`
- `constructor_wins_total`
- `driver_constructor_synergy`
- `track_complexity`
- `altitude_m`
- `tire_compound_start`
- `avg_pit_stops_circuit`
- `season_factor`
- `incident_rate_hist_norm`

Colunas explicitamente excluidas de X por leakage ou causalidade:

- `finish_position`
- `points`
- `race_points`
- `fastest_lap_race`
- `previous_position`
- `safety_car_flag`

## Fundamentacao Bibliografica

Esta decisao segue o mapa de referencias da arquitetura:

- Walk-forward validation e time-decay: Henderson et al. [9], Tan et al. [18].
- Coeficientes RAPM: Henderson et al. [9], Snoeks [10].
- Features de forma, sinergia, circuito e risco: Ruan et al. [2], Barra et al.
  [3], Heilmeier et al. [6].
- XGBoost: Chen & Guestrin [19].
- Random Forest: Breiman [20].
- LightGBM como terceiro candidato empirico: Barra et al. [3].
- Otimizacao de hiperparametros: Bergstra & Bengio [22], Akiba et al. [23].
- Metricas e benchmarks: Polishchuk [1], Alonso et al. [4], Henderson et al. [9].

## Conclusao

A Fase 1 fica fechada com **LightGBM** e **Random Forest** como modelos
finalistas. O **XGBoost** permanece documentado como terceiro algoritmo avaliado,
mas nao sera priorizado na apresentacao. O **Ridge Regression** deve ser
apresentado como baseline forte e como evidencia de que os coeficientes inspirados
em RAPM capturam parte relevante da estrutura dos dados.
