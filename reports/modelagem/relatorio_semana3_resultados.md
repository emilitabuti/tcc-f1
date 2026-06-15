# Relatorio Semana 3 - Resultados e Visualizacoes

## Criterio oficial

A avaliacao oficial usa `finish_position` como target e as metricas MAE, RMSE, R2 e Kendall tau. Top-3 nao faz parte do criterio oficial.

## Resultado consolidado

- Melhor modelo global por score composto: Ridge (score=0.5314, MAE=2.2723).
- Melhor modelo de arvore: LightGBM (score=0.5279, MAE=2.3172).
- Random Forest e LightGBM ficam muito proximos; XGBoost permanece como comparativo relevante da literatura.

## Figuras geradas

- `reports/modelagem/figures/semana3/01_mae_medio.png`
- `reports/modelagem/figures/semana3/02_rmse_medio.png`
- `reports/modelagem/figures/semana3/03_r2_medio.png`
- `reports/modelagem/figures/semana3/04_kendall_tau_medio.png`
- `reports/modelagem/figures/semana3/05_score_composto.png`
- `reports/modelagem/figures/semana3/06_mae_por_fold.png`
- `reports/modelagem/figures/semana3/07_rmse_por_fold.png`
- `reports/modelagem/figures/semana3/08_r2_por_fold.png`
- `reports/modelagem/figures/semana3/09_kendall_por_fold.png`
- `reports/modelagem/figures/semana3/10_feature_importance_lightgbm.png`
- `reports/modelagem/figures/semana3/10_feature_importance_random_forest.png`
- `reports/modelagem/figures/semana3/10_feature_importance_xgboost.png`

## Features dominantes

- LightGBM: qualifying_position, constructor_coef_rapm, recent_form_5, driver_constructor_synergy, track_complexity.
- Random Forest: qualifying_position, constructor_coef_rapm, recent_form_5, driver_constructor_synergy, constructor_wins_total.
- XGBoost: qualifying_position, constructor_coef_rapm, recent_form_5, driver_constructor_synergy, constructor_wins_total.

## Artefatos tabulares

- `reports/modelagem/tabela_metricas_tunadas_4modelos_resumo.csv`
- `reports/modelagem/tabela_metricas_tunadas_4modelos.csv`
- `reports/modelagem/tabela_features_dominantes_semana3.csv`
