# Decisao Preliminar dos Algoritmos Finalistas

## Resultado

Finalistas preliminares para a Fase 1:

- lightgbm_tuned: score composto 0.5279, MAE medio 2.3172, Kendall tau 0.6536, MAE std 0.1233.
- random_forest_tuned: score composto 0.5272, MAE medio 2.3263, Kendall tau 0.6503, MAE std 0.1220.

Modelo arquivado como terceiro candidato:

- xgboost_tuned: score composto 0.5269, MAE medio 2.3415, Kendall tau 0.6525, MAE std 0.1448.

Baseline linear:

- ridge_baseline: score composto 0.5314, MAE medio 2.2723, Kendall tau 0.6543. Permanece como referencia metodologica, nao como finalista principal.

## Justificativa

A escolha segue o criterio multi-metrica definido na revisao: equilibrar MAE, RMSE, R2 e Kendall tau, mantendo estabilidade entre folds e coerencia com a arquitetura. O baseline Ridge foi mantido como referencia linear forte baseada na fundamentacao RAPM; como ele ficou competitivo, os modelos de arvore devem ser justificados tambem pela analise de importancia de features, robustez e uso posterior na Fase 2 de drift/adaptacao.

A decisao ainda deve ser confirmada apos a etapa de feature selection e feature importance.