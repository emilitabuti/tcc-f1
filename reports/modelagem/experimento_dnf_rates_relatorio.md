# Experimento DNF Rates - Ablacao de Features

## Objetivo

Avaliar se as features `driver_dnf_rate` e `constructor_dnf_rate` melhoram a predicao de `finish_position` no dataset final DNF Excluded. O teste remove uma ou ambas as features e compara as metricas walk-forward contra a configuracao atual.

## Configuracao

- Modelo: XGBoost com hiperparametros ja tunados em `C:\Users\isagr\Documents\tcc-f1\reports\modelagem\optuna_xgboost_best_params.json`.
- X: `C:\Users\isagr\Documents\tcc-f1\data\processed\dataset_modelagem_X_2018_2025.csv`.
- y/metadados: `C:\Users\isagr\Documents\tcc-f1\data\processed\dataset_modelagem_y_2018_2025.csv`.
- Time-decay: `0.95`.
- Folds: treino ate 2022 -> 2023, treino ate 2023 -> 2024, treino ate 2024 -> 2025.
- Target: `finish_position`.
- Metricas: MAE, RMSE, R2, Kendall tau medio por corrida e Top-3 accuracy.

## Cenarios

- A - `base_15_features`: Dataset final atual, com driver_dnf_rate e constructor_dnf_rate.
- B - `sem_driver_dnf_rate`: Remove apenas a taxa historica de DNF atribuida ao piloto.
- C - `sem_constructor_dnf_rate`: Remove apenas a taxa historica de DNF mecanico do construtor.
- D - `sem_duas_dnf_rates`: Remove simultaneamente as duas taxas de DNF.

## Parametros do Modelo

```json
{
  "colsample_bytree": 0.6544778704537475,
  "learning_rate": 0.02207743518025897,
  "max_depth": 3,
  "n_estimators": 269,
  "reg_alpha": 0.7219110545028448,
  "reg_lambda": 0.5782115773391785,
  "subsample": 0.6320667544387549
}
```

## Resumo Medio

| experimento | cenario | n_features | mae_medio | delta_mae_medio_vs_base | rmse_medio | r2_medio | kendall_tau_medio | top3_accuracy_medio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B | sem_driver_dnf_rate | 14 | 2.322897 | -0.023486 | 3.002933 | 0.660809 | 0.652198 | 0.241162 |
| D | sem_duas_dnf_rates | 13 | 2.335588 | -0.010796 | 3.022909 | 0.656162 | 0.647944 | 0.241162 |
| C | sem_constructor_dnf_rate | 14 | 2.335602 | -0.010781 | 3.014760 | 0.657992 | 0.649703 | 0.227273 |
| A | base_15_features | 15 | 2.346383 | 0.000000 | 3.022463 | 0.656285 | 0.652334 | 0.241162 |

## Resultados por Fold

| experimento | cenario | valid_season | n_features | mae | rmse | r2 | kendall_tau | top3_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | base_15_features | 2023 | 15 | 2.451036 | 3.090837 | 0.630593 | 0.644320 | 0.181818 |
| A | base_15_features | 2024 | 15 | 2.174837 | 2.859529 | 0.701347 | 0.685416 | 0.166667 |
| A | base_15_features | 2025 | 15 | 2.413276 | 3.117024 | 0.636914 | 0.627267 | 0.375000 |
| B | sem_driver_dnf_rate | 2023 | 14 | 2.426149 | 3.063896 | 0.637005 | 0.644468 | 0.181818 |
| B | sem_driver_dnf_rate | 2024 | 14 | 2.159057 | 2.851669 | 0.702987 | 0.677824 | 0.166667 |
| B | sem_driver_dnf_rate | 2025 | 14 | 2.383485 | 3.093234 | 0.642435 | 0.634303 | 0.375000 |
| C | sem_constructor_dnf_rate | 2023 | 14 | 2.441188 | 3.076970 | 0.633900 | 0.641279 | 0.181818 |
| C | sem_constructor_dnf_rate | 2024 | 14 | 2.165813 | 2.845772 | 0.704214 | 0.682891 | 0.166667 |
| C | sem_constructor_dnf_rate | 2025 | 14 | 2.399805 | 3.121538 | 0.635862 | 0.624939 | 0.333333 |
| D | sem_duas_dnf_rates | 2023 | 13 | 2.443286 | 3.091932 | 0.630331 | 0.643018 | 0.181818 |
| D | sem_duas_dnf_rates | 2024 | 13 | 2.167368 | 2.857507 | 0.701769 | 0.677622 | 0.166667 |
| D | sem_duas_dnf_rates | 2025 | 13 | 2.396109 | 3.119289 | 0.636386 | 0.623191 | 0.375000 |

## Interpretacao

O melhor MAE medio foi do cenario B (`sem_driver_dnf_rate`), com MAE medio=2.322897. A configuracao base teve MAE medio=2.346383.
A diferenca e pequena e deve ser lida como evidencia de sinal fraco, nao como prova definitiva de que as taxas DNF devam ser removidas. Como o dataset de modelagem exclui DNFs, essas taxas historicas tendem a explicar pouco a posicao final entre pilotos que terminaram/classificaram a corrida.
Metodologicamente, manter as features continua defensavel porque elas sao causais e baseadas no historico classificado completo de DNF. Para desempenho puro, porem, o teste sugere que `driver_dnf_rate` e uma candidata razoavel a ablação em versoes futuras do modelo.

## Artefatos Gerados

- `C:\Users\isagr\Documents\tcc-f1\reports\modelagem\experimento_dnf_rates_metricas.csv`
- `C:\Users\isagr\Documents\tcc-f1\reports\modelagem\experimento_dnf_rates_relatorio.md`
