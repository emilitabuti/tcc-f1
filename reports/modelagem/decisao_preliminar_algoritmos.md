# Decisao Preliminar dos Algoritmos Finalistas

## Resultado

Finalistas preliminares para a Fase 1:

- lightgbm_tuned: MAE medio 2.3129, Kendall tau 0.6532, MAE std 0.1126.
- random_forest_tuned: MAE medio 2.3251, Kendall tau 0.6497, MAE std 0.1210.

Modelo arquivado como terceiro candidato:

- xgboost_tuned: MAE medio 2.3409, Kendall tau 0.6507, MAE std 0.1458.

Baseline linear:

- ridge_baseline: MAE medio 2.2734, Kendall tau 0.6546. Permanece como referencia metodologica, nao como finalista principal.

## Justificativa

A escolha segue o criterio definido no cronograma revisado: menor MAE medio, maior Kendall tau, estabilidade entre folds e coerencia com a arquitetura. O baseline Ridge foi mantido como referencia linear forte baseada na fundamentacao RAPM; como ele ficou competitivo, os modelos de arvore devem ser justificados tambem pela analise de importancia de features, robustez e uso posterior na Fase 2 de drift/adaptacao.

A decisao ainda deve ser confirmada apos a etapa de feature selection e feature importance.