# Relatorio de Estudos de Ablacao

Baseline oficial atual: LightGBM e XGBoost com 13 features e decay=0.99.

## Melhores por score composto

| grupo              | experimento                  | modelo   |   mae_medio |   rmse_medio |   r2_medio |   kendall_tau_medio |   top3_accuracy_medio |   score_composto |
|:-------------------|:-----------------------------|:---------|------------:|-------------:|-----------:|--------------------:|----------------------:|-----------------:|
| target             | target_delta_grid_to_finish  | LightGBM |     2.28335 |      3.04743 |   0.649801 |            0.647191 |              0.299242 |         0.503016 |
| ensemble           | media_arvores                | ensemble |     2.34003 |      3.01646 |   0.657951 |            0.651549 |              0.29798  |         0.502813 |
| decay_sem_retuning | decay_0.95                   | LightGBM |     2.32265 |      3.01305 |   0.658595 |            0.649988 |              0.284091 |         0.501139 |
| loss               | lgb_objective_regression_l1  | LightGBM |     2.2929  |      3.02767 |   0.655392 |            0.649306 |              0.270202 |         0.499348 |
| decay_sem_retuning | decay_0.97                   | XGBoost  |     2.34991 |      3.02201 |   0.656543 |            0.653535 |              0.270202 |         0.498387 |
| target             | target_delta_grid_to_finish  | XGBoost  |     2.2829  |      3.05228 |   0.648228 |            0.644176 |              0.270202 |         0.49817  |
| features           | baseline_13                  | LightGBM |     2.32636 |      3.01461 |   0.658227 |            0.652999 |              0.256313 |         0.497122 |
| decay_sem_retuning | decay_0.99                   | LightGBM |     2.32636 |      3.01461 |   0.658227 |            0.652999 |              0.256313 |         0.497122 |
| features           | add_incident_rate_hist_norm  | LightGBM |     2.33001 |      3.01277 |   0.658623 |            0.652063 |              0.256313 |         0.496986 |
| features           | remove_avg_pit_stops_circuit | XGBoost  |     2.33611 |      3.01266 |   0.658607 |            0.653468 |              0.256313 |         0.496961 |
| decay_sem_retuning | decay_0.99                   | XGBoost  |     2.34795 |      3.02069 |   0.656875 |            0.652344 |              0.256313 |         0.496283 |
| features           | baseline_13                  | XGBoost  |     2.34795 |      3.02069 |   0.656875 |            0.652344 |              0.256313 |         0.496283 |

## Melhores por RMSE

| grupo              | experimento                  | modelo   |   mae_medio |   rmse_medio |   r2_medio |   kendall_tau_medio |   top3_accuracy_medio |   score_composto |
|:-------------------|:-----------------------------|:---------|------------:|-------------:|-----------:|--------------------:|----------------------:|-----------------:|
| ensemble           | ridge_70_xgb_30              | ensemble |     2.28385 |      2.95826 |   0.670847 |            0.654639 |              0.214646 |         0.493997 |
| ensemble           | ridge_70_lgb_30              | ensemble |     2.27774 |      2.95865 |   0.67074  |            0.65543  |              0.214646 |         0.494232 |
| ensemble           | ridge_50_lgb_50              | ensemble |     2.28595 |      2.96714 |   0.668861 |            0.654508 |              0.214646 |         0.493642 |
| features           | remove_avg_pit_stops_circuit | LightGBM |     2.31935 |      3.01074 |   0.658903 |            0.651605 |              0.227273 |         0.49292  |
| decay_sem_retuning | decay_1.00                   | LightGBM |     2.32495 |      3.01145 |   0.658964 |            0.651661 |              0.241162 |         0.494857 |
| decay_sem_retuning | decay_0.98                   | LightGBM |     2.32528 |      3.01197 |   0.658849 |            0.651857 |              0.241162 |         0.494851 |
| features           | remove_avg_pit_stops_circuit | XGBoost  |     2.33611 |      3.01266 |   0.658607 |            0.653468 |              0.256313 |         0.496961 |
| features           | add_incident_rate_hist_norm  | LightGBM |     2.33001 |      3.01277 |   0.658623 |            0.652063 |              0.256313 |         0.496986 |
| ensemble           | lgb_70_xgb_30                | ensemble |     2.33072 |      3.01303 |   0.658612 |            0.65243  |              0.242424 |         0.494917 |
| decay_sem_retuning | decay_0.95                   | LightGBM |     2.32265 |      3.01305 |   0.658595 |            0.649988 |              0.284091 |         0.501139 |
| ensemble           | media_lgb_xgb                | ensemble |     2.33493 |      3.0136  |   0.658493 |            0.652907 |              0.242424 |         0.494833 |
| features           | baseline_13                  | LightGBM |     2.32636 |      3.01461 |   0.658227 |            0.652999 |              0.256313 |         0.497122 |

## Melhores por top-3

| grupo               | experimento                        | modelo        |   mae_medio |   rmse_medio |   r2_medio |   kendall_tau_medio |   top3_accuracy_medio |   score_composto |
|:--------------------|:-----------------------------------|:--------------|------------:|-------------:|-----------:|--------------------:|----------------------:|-----------------:|
| target              | target_delta_grid_to_finish        | LightGBM      |     2.28335 |      3.04743 |   0.649801 |            0.647191 |              0.299242 |         0.503016 |
| loss                | xgb_objective_reg:pseudohubererror | XGBoost       |    22.5747  |     23.1578  | -19.1253   |          nan        |              0.299242 |         0.263821 |
| ensemble            | media_arvores                      | ensemble      |     2.34003 |      3.01646 |   0.657951 |            0.651549 |              0.29798  |         0.502813 |
| decay_sem_retuning  | decay_0.95                         | LightGBM      |     2.32265 |      3.01305 |   0.658595 |            0.649988 |              0.284091 |         0.501139 |
| loss                | lgb_objective_regression_l1        | LightGBM      |     2.2929  |      3.02767 |   0.655392 |            0.649306 |              0.270202 |         0.499348 |
| decay_sem_retuning  | decay_0.97                         | XGBoost       |     2.34991 |      3.02201 |   0.656543 |            0.653535 |              0.270202 |         0.498387 |
| target              | target_delta_grid_to_finish        | XGBoost       |     2.2829  |      3.05228 |   0.648228 |            0.644176 |              0.270202 |         0.49817  |
| podio_classificador | XGBClassifier                      | XGBClassifier |   nan       |    nan       | nan        |          nan        |              0.270202 |       nan        |
| loss                | xgb_objective_reg:absoluteerror    | XGBoost       |     2.33177 |      3.03525 |   0.653704 |            0.642612 |              0.256313 |         0.495293 |
| features            | remove_season_factor               | XGBoost       |     2.35585 |      3.03275 |   0.65405  |            0.650512 |              0.256313 |         0.495495 |
| decay_sem_retuning  | decay_1.00                         | XGBoost       |     2.35179 |      3.02515 |   0.655858 |            0.650188 |              0.256313 |         0.495822 |
| features            | baseline_13                        | XGBoost       |     2.34795 |      3.02069 |   0.656875 |            0.652344 |              0.256313 |         0.496283 |

## Observacoes

- Experimentos de features, decay, loss e target usam os hiperparametros atuais, sem retuning completo.
- Experimentos promissores devem ser retunados antes de virar decisao oficial.
- O classificador de podium avalia apenas top-3, pois nao produz predicao numerica de posicao.

## Retuning do candidato target_delta_grid_to_finish

Depois da triagem inicial, o candidato `target_delta_grid_to_finish` foi retunado com Optuna para LightGBM e XGBoost pelo script `src/tuning_target_delta_ablacao.py`.

Resultados gerados em `reports/ablacao/resultados_target_delta_retuned.csv`:

| experimento | modelo | mae_medio | rmse_medio | r2_medio | kendall_tau_medio | top3_accuracy_medio | score_composto |
|:---|:---|---:|---:|---:|---:|---:|---:|
| XGBoost_target_delta_retuned | XGBoost | 2.265735 | 3.003344 | 0.660137 | 0.657548 | 0.257576 | 0.499737 |
| LightGBM_target_delta_retuned | LightGBM | 2.279983 | 3.022421 | 0.656155 | 0.651259 | 0.257576 | 0.498133 |

Conclusao: o melhor ganho confirmado apos retuning foi o `XGBoost_target_delta_retuned`. Ele supera os baselines oficiais de LightGBM e XGBoost no score composto e melhora principalmente MAE, RMSE, R2 e Kendall. O ganho de top-3 da triagem inicial nao se manteve apos retuning, ficando em 25.8%.
