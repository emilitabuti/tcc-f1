# Experimento de Janela Historica de Treino - Teste 2025

## Objetivo

Avaliar se reduzir progressivamente o inicio da janela historica de treino provoca queda nas metricas de validacao em 2025. O teste mantem fixos o modelo, o ano de teste, as features finais, os hiperparametros e o time-decay; apenas o primeiro ano incluido no treino muda.

## Configuracao

- Modelo: LightGBM tunado, versao `lightgbm==4.6.0`.
- X: `C:\Users\isagr\Documents\tcc-f1\data\processed\dataset_modelagem_X_2018_2025.csv`.
- y/metadados: `C:\Users\isagr\Documents\tcc-f1\data\processed\dataset_modelagem_y_2018_2025.csv`.
- Parametros: `C:\Users\isagr\Documents\tcc-f1\reports\modelagem\optuna_lightgbm_best_params.json`.
- Time-decay: `0.95`.
- Teste fixo: `2025`.
- Target: `finish_position`.
- Metricas: MAE, R2, Top-3 accuracy, Kendall tau e RMSE.

## Parametros do Modelo

```json
{
  "colsample_bytree": 0.8625417840147478,
  "learning_rate": 0.03164237033214078,
  "max_depth": 4,
  "min_child_samples": 32,
  "n_estimators": 146,
  "num_leaves": 8,
  "reg_alpha": 0.805684044053038,
  "reg_lambda": 0.5585100010421958,
  "subsample": 0.9926137770798064
}
```

## Resultados

| experimento | anos_de_treino | n_train | mae | delta_mae_vs_a | r2 | delta_r2_vs_a | top3_accuracy | delta_top3_vs_a | kendall_tau | rmse |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 2018-2024 | 2524 | 2.357896 | 0.000000 | 0.647683 | 0.000000 | 0.291667 | 0.000000 | 0.641194 | 3.070451 |
| B | 2019-2024 | 2189 | 2.353329 | -0.004567 | 0.650487 | 0.002804 | 0.291667 | 0.000000 | 0.639007 | 3.058206 |
| C | 2020-2024 | 1829 | 2.368628 | 0.010733 | 0.645062 | -0.002621 | 0.291667 | 0.000000 | 0.637205 | 3.081850 |
| D | 2021-2024 | 1546 | 2.381465 | 0.023570 | 0.642896 | -0.004787 | 0.250000 | -0.041667 | 0.638126 | 3.091241 |

Os deltas foram calculados em relacao ao experimento A. Para MAE, valor positivo indica piora; para R2 e Top-3 accuracy, valor negativo indica piora.

## Leitura Objetiva

- Melhor MAE: experimento B (2019-2024), MAE=2.353329.
- Melhor R2: experimento B (2019-2024), R2=0.650487.
- Melhor Top-3 accuracy: experimento A (2018-2024), Top-3=0.291667.

## Observacao Metodologica

Este experimento testa o efeito da quantidade de linhas usadas no ajuste do modelo. As features ja estavam previamente calculadas pelo pipeline final do projeto; portanto, o teste nao reexecuta toda a engenharia de features para cada janela. Essa escolha e coerente com a pergunta operacional: mantido o dataset final, vale treinar com 2018-2024 ou remover anos antigos?

## Artefatos Gerados

- `C:\Users\isagr\Documents\tcc-f1\reports\modelagem\experimento_janela_treino_2025_metricas.csv`
- `C:\Users\isagr\Documents\tcc-f1\reports\modelagem\experimento_janela_treino_2025_predicoes.csv`
- `C:\Users\isagr\Documents\tcc-f1\reports\modelagem\experimento_janela_treino_2025_relatorio.md`
