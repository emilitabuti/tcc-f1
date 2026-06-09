# Decisao Final dos Algoritmos - Fase 1

Data de fechamento: 30/05/2026

## Resultado

Algoritmos finalistas para apresentacao da Fase 1:

1. **LightGBM**
2. **XGBoost**

Algoritmo de arvore arquivado:

- **Random Forest**

Baseline mantido como referencia metodologica:

- **Ridge Regression**

## Evidencias Quantitativas

| Modelo | Score composto | MAE medio | MAE DP | RMSE medio | R2 medio | Kendall tau | Top-3 accuracy | Tempo tuning |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LightGBM | 0.4971 | 2.3264 | 0.1316 | 3.0146 | 0.6582 | 0.6530 | 0.2563 | 0.37 min |
| XGBoost | 0.4963 | 2.3479 | 0.1358 | 3.0207 | 0.6569 | 0.6523 | 0.2563 | 0.61 min |
| Random Forest | 0.4957 | 2.3732 | 0.1235 | 3.0515 | 0.6501 | 0.6436 | 0.2689 | 1.14 min |
| Ridge baseline | 0.4900 | 2.2723 | 0.1306 | 2.9574 | 0.6710 | 0.6543 | 0.1856 | 0.07 min |

Arquivos de origem:

- `reports/modelagem/tabela_metricas_tunadas_4modelos.csv`
- `reports/modelagem/tabela_metricas_tunadas_4modelos_resumo.csv`

## Justificativa da Escolha

O criterio revisado foi aplicado de forma empirica por score composto multi-metrica:

- MAE invertido: peso 0.30;
- RMSE invertido: peso 0.15;
- R2: peso 0.20;
- Kendall tau: peso 0.20;
- top-3 accuracy: peso 0.15.

O **LightGBM** apresentou o melhor score composto medio (`0.4971`) e menor tempo
de tuning entre os modelos de arvore finalistas. O **XGBoost** ficou em
segundo no score composto (`0.4963`) e manteve o mesmo top-3 medio do
LightGBM.

O **Random Forest** foi arquivado como terceiro candidato de arvore. Seu top-3
medio foi o maior entre as arvores, mas o score composto ficou abaixo de
LightGBM e XGBoost porque MAE, RMSE, R2 e Kendall tau foram piores.

O **Ridge baseline** obteve o menor MAE, menor RMSE e maior R2. Esse resultado
deve ser reportado explicitamente, pois mostra que a decomposicao linear inspirada
em RAPM e forte para este dataset. Ainda assim, ele permanece como baseline,
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

A Fase 1 fica fechada com **LightGBM** e **XGBoost** como modelos finalistas.
O **Random Forest** permanece documentado como terceiro algoritmo avaliado, mas
nao sera priorizado na apresentacao pelo criterio multi-metrica. O **Ridge Regression** deve ser apresentado
como baseline forte e como evidencia de que os coeficientes inspirados em RAPM
capturam parte relevante da estrutura dos dados.
