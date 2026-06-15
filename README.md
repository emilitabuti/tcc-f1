# TCC F1 - Predicao de Resultado de Corridas

Projeto de TCC para prever `finish_position` em corridas de Formula 1 com validacao temporal walk-forward, features pre-corrida e comparacao de modelos de regressao/ranking.

## Objetivo

O pipeline principal estima a posicao final de pilotos antes da corrida, usando somente informacoes disponiveis previamente ou calculadas com historico anterior. A formulacao oficial e regressao causal de `finish_position`.

Metricas oficiais:

- MAE;
- RMSE;
- R2;
- Kendall tau medio por corrida.

Top-3 nao faz parte do criterio oficial. Experimentos ou arquivos antigos com top-3 devem ser tratados apenas como historico.

## Estrutura

| Pasta | Conteudo |
|---|---|
| `src/` | Scripts de coleta, tratamento, feature engineering, modelagem e visualizacao |
| `data/raw/` | Dados brutos Ergast/Jolpica, FastF1 e OpenF1 |
| `data/processed/` | Bases intermediarias e datasets finais de modelagem |
| `models/` | Artefatos de preprocessing, RAPM e selecao de features |
| `reports/` | Metricas, predicoes, graficos e relatorios gerados |
| `docs/` | Documentacao tecnica, inventarios e decisoes metodologicas |

## Ambiente

Instalar dependencias:

```bash
python3 -m pip install -r requirements.txt
```

Versoes principais estao fixadas em `requirements.txt`, incluindo pandas, scikit-learn, XGBoost, LightGBM, Optuna, matplotlib e seaborn.

## Dados Finais

Artefatos centrais:

- `data/processed/dataset_modelagem_X_2018_2025.csv`;
- `data/processed/dataset_modelagem_y_2018_2025.csv`;
- `models/feature_selection/features_modelagem_2018_2025.json`.

O arquivo JSON de features e o contrato canonico das 13 features usadas na modelagem.

## Ordem de Execucao

Pipeline completo de dados:

```bash
python3 src/pipeline_dados.py
```

Validacao de schema 2025:

```bash
python3 src/validar_schema_2025_modelagem.py
```

Otimizacao de time-decay:

```bash
python3 src/otimizacao_time_decay.py
```

Tuning e baseline:

```bash
python3 src/tuning_xgboost.py
python3 src/tuning_randomforest.py
python3 src/tuning_lightgbm.py
python3 src/otimizacao_ridge_lambda.py
```

Consolidacao de metricas:

```bash
python3 src/consolidar_metricas_tunadas.py
```

Feature importance:

```bash
python3 src/gerar_feature_importance_modelos.py
```

Visualizacoes finais da Semana 3:

```bash
python3 src/gerar_visualizacoes_semana3.py
```

Atualizacao exploratoria 2026:

```bash
python3 src/update_openf1_2026.py
python3 src/avaliar_2026_semana3.py
```

## Resultados Oficiais

Tabela principal:

- `reports/modelagem/tabela_metricas_tunadas_4modelos_resumo.csv`;
- `reports/modelagem/tabela_metricas_tunadas_4modelos.csv`.

Resumo atual:

| Modelo | Leitura |
|---|---|
| Ridge | Melhor modelo/baseline global no setup oficial |
| LightGBM | Melhor modelo de arvore |
| Random Forest | Arvore robusta, muito proxima do LightGBM |
| XGBoost | Comparativo relevante, arquivado como terceiro candidato de arvore |

Relatorios principais:

- `reports/modelagem/decisao_algoritmos.md`;
- `reports/modelagem/relatorio_semana3_resultados.md`;
- `reports/modelagem/validacao_2026_semana3.md`.
- `reports/modelagem/analise_2026_semana3.md`.

Graficos finais:

- `reports/modelagem/figures/semana3/`.
- `reports/modelagem/figures/semana3_2026/`.

Notebook demonstrativo:

- `notebooks/notebook_demonstracao_fase1.ipynb`.

## Ablacoes

O plano oficial de ablacooes esta em:

- `docs/tecnico/11_plano_estudos_ablacao.md`.

Somente ablacooes com `target_mode=finish` devem ser usadas para decisao metodologica. Transformacoes como `delta_grid`, `rank_norm_grid20` ou `log1p_finish` sao historicas e nao substituem o target oficial.

## Semana 3

O plano revisado da Semana 3 esta em:

- `docs/tecnico/14_plano_execucao_semana3_revisado.md`.

A Semana 3 consolida documentacao, reproducao, notebook demonstrativo, visualizacoes e conferencia entre codigo e texto. Ela nao deve abrir nova decisao ampla de modelagem sem justificativa.
