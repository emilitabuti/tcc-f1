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

| Modelo | Score composto | MAE medio | MAE DP | RMSE medio | R2 medio | Kendall tau | Tempo tuning |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ridge baseline | 0.5314 | 2.2723 | 0.1306 | 2.9574 | 0.6710 | 0.6543 | 0.13 min |
| LightGBM | 0.5279 | 2.3172 | 0.1233 | 3.0121 | 0.6587 | 0.6536 | 3.88 min |
| Random Forest | 0.5272 | 2.3263 | 0.1220 | 3.0121 | 0.6589 | 0.6503 | 4.01 min |
| XGBoost | 0.5269 | 2.3415 | 0.1448 | 3.0161 | 0.6578 | 0.6525 | 4.19 min |

Arquivos de origem:

- `reports/modelagem/tabela_metricas_tunadas_4modelos.csv`
- `reports/modelagem/tabela_metricas_tunadas_4modelos_resumo.csv`

## Justificativa da Escolha

O criterio revisado foi aplicado de forma empirica por score composto multi-metrica:

- MAE invertido: peso 0.35;
- RMSE invertido: peso 0.20;
- R2: peso 0.20;
- Kendall tau: peso 0.25.

O **LightGBM** apresentou o melhor score composto medio entre os modelos de
arvore (`0.5279`) e o menor MAE medio das arvores. O **Random Forest** ficou em
segundo entre as arvores (`0.5272`), com RMSE e R2 praticamente empatados com
LightGBM e maior estabilidade de MAE.

O **XGBoost** foi arquivado como terceiro candidato de arvore. A diferenca para
Random Forest e pequena (`0.5269` vs. `0.5272`), mas o score composto revisado,
sem top-3, favoreceu Random Forest.

O **Ridge baseline** obteve o melhor score composto, menor MAE, menor RMSE,
maior R2 e maior Kendall tau. Esse resultado deve ser reportado explicitamente,
pois mostra que a decomposicao linear inspirada em RAPM e forte para este dataset.
Ainda assim, ele permanece como baseline,
porque o objetivo da Fase 1 inclui comparar modelos de arvore interpretaveis e
preparar a analise de drift/adaptacao da Fase 2.

## Feature Importance

Top features por modelo:

| Modelo | Top 5 features |
|---|---|
| XGBoost | `qualifying_position`, `constructor_coef_rapm`, `recent_form_5`, `driver_constructor_synergy`, `constructor_wins_total` |
| Random Forest | `qualifying_position`, `constructor_coef_rapm`, `recent_form_5`, `driver_constructor_synergy`, `constructor_wins_total` |
| LightGBM | `qualifying_position`, `constructor_coef_rapm`, `recent_form_5`, `driver_constructor_synergy`, `constructor_wins_total` |

Arquivos gerados:

- `reports/modelagem/feature_importance_xgb.csv`
- `reports/modelagem/feature_importance_rf.csv`
- `reports/modelagem/feature_importance_lgb.csv`
- `reports/modelagem/feature_importance_2024.csv`
- `reports/modelagem/relatorio_feature_importance_29_30_05.txt`

## Dataset Final

O dataset de modelagem foi validado com:

- 2.943 linhas;
- 13 features em X;
- 0 valores NaN em X e y;
- temporadas 2018 a 2025;
- nenhuma coluna proibida em X.

Features finais:

- `qualifying_position`
- `recent_form_5`
- `constructor_coef_rapm`
- `driver_constructor_synergy`
- `constructor_wins_total`
- `driver_coef_rapm`
- `season_factor`
- `tire_compound_start`
- `avg_pit_stops_circuit`
- `track_complexity`
- `grid_penalty`
- `constructor_dnf_rate`
- `altitude_m`

## Conclusao

A Fase 1 fica fechada com **LightGBM** e **Random Forest** como modelos finalistas.
O **XGBoost** permanece documentado como terceiro algoritmo avaliado, mas
nao sera priorizado na apresentacao pelo criterio multi-metrica. O **Ridge Regression** deve ser apresentado
como baseline forte e como evidencia de que os coeficientes inspirados em RAPM
capturam parte relevante da estrutura dos dados.
