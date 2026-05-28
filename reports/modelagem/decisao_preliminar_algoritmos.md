# Decisao Preliminar dos Algoritmos Finalistas

## Resultado

Finalistas preliminares para a Fase 1:

- lightgbm_tuned: MAE medio 2.3133, Kendall tau 0.6551, MAE std 0.1117.
- random_forest_tuned: MAE medio 2.3275, Kendall tau 0.6511, MAE std 0.1210.

Modelo arquivado como terceiro candidato:

- xgboost_tuned: MAE medio 2.3342, Kendall tau 0.6518, MAE std 0.1334.

Baseline linear:

- ridge_baseline: MAE medio 2.2734, Kendall tau 0.6546. Permanece como referencia metodologica, nao como finalista principal.

## Justificativa

A escolha segue o criterio definido no cronograma revisado: menor MAE medio, maior Kendall tau, estabilidade entre folds e coerencia com a arquitetura. O baseline Ridge foi mantido como referencia linear forte baseada na fundamentacao RAPM; como ele ficou competitivo, os modelos de arvore devem ser justificados tambem pela analise de importancia de features, robustez e uso posterior na Fase 2 de drift/adaptacao.

A decisao ainda deve ser confirmada apos a etapa de feature selection e feature importance.