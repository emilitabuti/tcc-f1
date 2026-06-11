# Ablação pareada LightGBM vs XGBoost

## Conclusão Executiva

- Melhor configuração por score oficial: LightGBM com target oficial fixo `finish_position`, `decay=0.99` e `score_profile=atual`.
- Métricas médias: MAE 2.3226, RMSE 3.0075, R2 0.6599, Kendall tau 0.6528, score 0.5278.
- O alvo nao foi transformado: todas as configuracoes mantem `target_mode=finish`.
- Como este estudo compara apenas LightGBM e XGBoost, ele orienta o próximo retuning; não substitui sozinho a tabela final com todos os modelos.

## Protocolo

- Todos os modelos foram avaliados com as mesmas configurações experimentais: mesmos folds, mesmos targets, mesmos fatores de time-decay, mesmos perfis de score e mesmo número de trials.
- A única diferença é o espaço de hiperparâmetros próprio de cada algoritmo.
- Folds de tuning: 2023 e 2024.
- Folds de avaliação final: 2023, 2024 e 2025.
- Trials por combinação e modelo: 20.
- Target oficial mantido: finish.
- Decays testados: 0.95, 0.99.
- Perfis de score testados: atual, rmse_r2, erro_continuo.
- Transformacoes do alvo, como `delta_grid` ou `rank_norm_grid20`, nao fazem parte do escopo oficial.

## Melhores resultados por score oficial

| modelo   | target_mode   |   decay | score_profile   |   mae_medio |   rmse_medio |   r2_medio |   kendall_tau_medio |   score_composto |   score_perfil_avaliacao |
|:---------|:--------------|--------:|:----------------|------------:|-------------:|-----------:|--------------------:|-----------------:|-------------------------:|
| LightGBM | finish        |  0.9900 | atual           |      2.3226 |       3.0075 |     0.6599 |              0.6528 |           0.5278 |                   0.5278 |
| LightGBM | finish        |  0.9900 | rmse_r2         |      2.3253 |       3.0083 |     0.6597 |              0.6530 |           0.5277 |                   0.5229 |
| LightGBM | finish        |  0.9900 | erro_continuo   |      2.3254 |       3.0100 |     0.6592 |              0.6525 |           0.5276 |                   0.4436 |
| XGBoost  | finish        |  0.9500 | atual           |      2.3445 |       3.0236 |     0.6561 |              0.6496 |           0.5262 |                   0.5262 |
| XGBoost  | finish        |  0.9500 | rmse_r2         |      2.3445 |       3.0236 |     0.6561 |              0.6496 |           0.5262 |                   0.5214 |
| XGBoost  | finish        |  0.9500 | erro_continuo   |      2.3445 |       3.0236 |     0.6561 |              0.6496 |           0.5262 |                   0.4423 |
| XGBoost  | finish        |  0.9900 | atual           |      2.3568 |       3.0287 |     0.6551 |              0.6494 |           0.5256 |                   0.5256 |
| XGBoost  | finish        |  0.9900 | rmse_r2         |      2.3568 |       3.0287 |     0.6551 |              0.6494 |           0.5256 |                   0.5209 |
| XGBoost  | finish        |  0.9900 | erro_continuo   |      2.3568 |       3.0287 |     0.6551 |              0.6494 |           0.5256 |                   0.4416 |
| LightGBM | finish        |  0.9500 | atual           |      2.3628 |       3.0592 |     0.6481 |              0.6426 |           0.5235 |                   0.5235 |
| LightGBM | finish        |  0.9500 | rmse_r2         |      2.3628 |       3.0592 |     0.6481 |              0.6426 |           0.5235 |                   0.5187 |
| LightGBM | finish        |  0.9500 | erro_continuo   |      2.3628 |       3.0592 |     0.6481 |              0.6426 |           0.5235 |                   0.4398 |

## Melhores por modelo

| modelo   | target_mode   |   decay | score_profile   |   mae_medio |   rmse_medio |   r2_medio |   kendall_tau_medio |   score_composto |   score_perfil_avaliacao |
|:---------|:--------------|--------:|:----------------|------------:|-------------:|-----------:|--------------------:|-----------------:|-------------------------:|
| LightGBM | finish        |  0.9900 | atual           |      2.3226 |       3.0075 |     0.6599 |              0.6528 |           0.5278 |                   0.5278 |
| XGBoost  | finish        |  0.9500 | atual           |      2.3445 |       3.0236 |     0.6561 |              0.6496 |           0.5262 |                   0.5262 |

## Configurações que batem as metas principais

Nenhuma configuração bateu simultaneamente RMSE < 3.0, R2 >= 0.66 e Kendall >= 0.60.

## Comparação pareada por configuração

| target_mode   |   decay | score_profile   |   LightGBM |   XGBoost |   delta_lgbm_menos_xgb | melhor_modelo_config   |
|:--------------|--------:|:----------------|-----------:|----------:|-----------------------:|:-----------------------|
| finish        |  0.9500 | atual           |     0.5235 |    0.5262 |                -0.0027 | XGBoost                |
| finish        |  0.9500 | erro_continuo   |     0.5235 |    0.5262 |                -0.0027 | XGBoost                |
| finish        |  0.9500 | rmse_r2         |     0.5235 |    0.5262 |                -0.0027 | XGBoost                |
| finish        |  0.9900 | atual           |     0.5278 |    0.5256 |                 0.0022 | LightGBM               |
| finish        |  0.9900 | erro_continuo   |     0.5276 |    0.5256 |                 0.0020 | LightGBM               |
| finish        |  0.9900 | rmse_r2         |     0.5277 |    0.5256 |                 0.0021 | LightGBM               |

## Artefatos

- Resultados consolidados: `/home/emili-tabuti/Documentos/projects/tcc-f1/reports/ablacao/pareada_lgbm_xgboost/resultados_pareados.csv`
- Relatório: `/home/emili-tabuti/Documentos/projects/tcc-f1/reports/ablacao/pareada_lgbm_xgboost/relatorio_pareado.md`
- Métricas, predições e parâmetros por configuração foram salvos no mesmo diretório.