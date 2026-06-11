# Inventário — models/ e reports/

Classificações: **Essencial** | **Importante** | **Temporário** | **Candidato à remoção**

---

## models/

Artefatos de modelos serializados e metadados de treinamento.

### models/preprocessing/

Scalers e encoders serializados. Necessários para reproduzir o pré-processamento exato.

| Arquivo | Classificação | Motivo |
|---|---|---|
| `standard_scaler_base_historica.joblib` | **Essencial** | Z-score ajustado na base histórica — usado pelo Ridge baseline |
| `standard_scaler_historico.joblib` | **Importante** | Versão alternativa do scaler — verificar se ainda é referenciada |
| `minmax_scaler_base_historica.joblib` | **Essencial** | MinMax para grid_position e laps |
| `minmax_scaler_historico.joblib` | **Importante** | Versão alternativa — verificar referências |
| `onehot_encoder_base_historica.joblib` | **Essencial** | OHE para circuito e construtor |
| `onehot_encoder_historico_fastf1.joblib` | **Importante** | OHE versão FastF1 — verificar se ainda é usado |
| `scaler_grid_position_fixed.joblib` | **Importante** | Scaler específico para grid_position com tratamento de zero |
| `schema_encoding_base_historica.json` | **Essencial** | Schema de encoding — valida que colunas OHE batem entre treino e validação |
| `schema_encoding_historico_fastf1.json` | **Importante** | Idem versão FastF1 |

### models/feature_selection/

Artefatos da seleção de features via RFE temporal.

| Arquivo | Classificação | Motivo |
|---|---|---|
| `features_modelagem_2018_2025.json` | **Essencial** | Contrato formal das 13 features — lista canônica usada em todo o pipeline de modelagem |
| `rfe_xgboost_ranking.csv` | **Essencial** | Ranking de importância por gain das 19 features candidatas — evidência da seleção |
| `rfe_xgboost_subsets.csv` | **Essencial** | Métricas e score composto de cada subconjunto testado no RFE — mostra por que 13 foi o número escolhido |
| `rfe_xgboost_pareto.csv` | **Essencial** | Subconjuntos Pareto-ótimos do RFE multi-métrica |
| `manifest_rfe_xgboost.json` | **Essencial** | Metadados da execução do RFE: parâmetros, datas, versões |
| `relatorio_rfe_xgboost.txt` | **Essencial** | Relatório legível da seleção: ranking final e subconjunto escolhido |
| `relatorio_13_selecao_features_modelagem.txt` | **Essencial** | Relatório da etapa 13: decisões de inclusão/exclusão de features |

### models/rapm/

| Arquivo | Classificação | Motivo |
|---|---|---|
| `manifest_rapm_ridge.json` | **Essencial** | Parâmetros do RAPM (alpha=10.0, decay=0.75, decay_unit=season), contagem de corridas processadas e contrato de merge |

---

## reports/

Resultados, visualizações e relatórios gerados pelas análises.

### reports/correlacao_features/

| Arquivo | Classificação | Motivo |
|---|---|---|
| `relatorio_correlacao.md` | **Essencial** | Relatório de correlação entre features candidatas/finais — inclui pares com r > 0.85 e correlação com o target |
| `relatorio_correlacao_features.txt` | **Importante** | Versão texto do mesmo relatório |
| `matriz_correlacao_features.csv` | **Essencial** | Matriz de correlação de Pearson completa entre as features analisadas |
| `correlation_with_target.csv` | **Essencial** | Correlação de cada feature com `finish_position` |
| `pares_correlacao_alta_maior_085.csv` | **Essencial** | Pares com \|r\| > 0.85 — base das decisões de remoção de multicolinearidade |
| `correlation_matrix_features.png` | **Importante** | Heatmap visual da matriz de correlação |

### reports/eda_dataset_tratado/

| Arquivo | Classificação | Motivo |
|---|---|---|
| `eda_dataset_tratado_summary.md` | **Essencial** | Sumário do EDA do dataset tratado |
| `ydata_profile_dataset_tratado.html` | **Importante** | Relatório completo de perfil do dataset (ydata-profiling) |
| `ydata_profile_dataset_tratado.json` | **Candidato à remoção** | Versão JSON do perfil — volumoso e o HTML já contém as mesmas informações |
| `figures/eda_principal/*.png` | **Importante** | 12 gráficos do EDA: distribuição do target, grid vs. finish, outliers, etc. |
| `great_expectations/checkpoint_result.json` | **Essencial** | Resultado da validação automática Great Expectations |
| `great_expectations/gx_core_validation_result.json` | **Essencial** | Resultado detalhado das validações de schema |
| `great_expectations/validation_summary.md` | **Essencial** | Resumo legível das validações |
| `gerar_graficos_eda_principais.py` | **Importante** | Script que gera os 12 gráficos do EDA — está em reports/ mas é um script de análise |
| `gerar_graficos_outliers_interpretaveis.py` | **Importante** | Script para gráficos de outliers |

### reports/modelagem/

#### Decisões e resultados finais

| Arquivo | Classificação | Motivo |
|---|---|---|
| `decisao_algoritmos.md` | **Essencial** | Decisão revisada: LightGBM + Random Forest finalistas de árvore, XGBoost arquivado como terceiro candidato |
| `decisao_preliminar_algoritmos.md` | **Candidato à remoção** | Versão anterior da decisão — substituída por `decisao_algoritmos.md` |
| `tabela_metricas_tunadas_4modelos.csv` | **Essencial** | Tabela definitiva: MAE por fold para os 4 modelos (LightGBM, RF, XGBoost, Ridge) |
| `tabela_metricas_tunadas_4modelos_resumo.csv` | **Essencial** | Resumo por modelo: MAE médio, DP, tempo de tuning |
| `feature_importance_2024.csv` | **Essencial** | Importância de features do fold 2024 — referência para análise de drift (antes da transição 2026) |
| `feature_importance_lgb.csv` | **Essencial** | Importância de features — LightGBM |
| `feature_importance_rf.csv` | **Essencial** | Importância de features — Random Forest |
| `feature_importance_xgb.csv` | **Essencial** | Importância de features — XGBoost |

#### Hiperparâmetros ótimos

| Arquivo | Classificação | Motivo |
|---|---|---|
| `optuna_lightgbm_best_params.json` | **Essencial** | Hiperparâmetros ótimos do LightGBM — necessários para reproduzir o modelo finalista |
| `optuna_randomforest_best_params.json` | **Essencial** | Idem Random Forest |
| `optuna_xgboost_best_params.json` | **Importante** | Idem XGBoost — modelo arquivado, mas manter para rastreabilidade |
| `ridge_best_params.json` | **Essencial** | Alpha ótimo do Ridge baseline |
| `optuna_lightgbm_trials.csv` | **Importante** | Todos os trials do Optuna — LightGBM |
| `optuna_randomforest_trials.csv` | **Importante** | Idem Random Forest |
| `optuna_xgboost_trials.csv` | **Importante** | Idem XGBoost |
| `ridge_alpha_grid.csv` | **Importante** | Grid de alphas testados no RidgeCV |

#### Time-decay

| Arquivo | Classificação | Motivo |
|---|---|---|
| `time_decay_escolhido_xgboost.txt` | **Essencial** | Decay=0.99 — valor escolhido por score composto e justificativa |
| `otimizacao_time_decay_xgboost.csv` | **Essencial** | Métricas por fator de decay testado (0.50 a 0.99) |
| `otimizacao_time_decay_xgboost_resumo.csv` | **Importante** | Resumo compacto da otimização |

#### Predições

| Arquivo | Classificação | Motivo |
|---|---|---|
| `predicoes_walk_forward_lightgbm_tuned.csv` | **Essencial** | Predições fold a fold do LightGBM tunado — finalista |
| `predicoes_walk_forward_randomforest_tuned.csv` | **Essencial** | Idem Random Forest tunado — finalista |
| `predicoes_walk_forward_xgboost_tuned.csv` | **Importante** | Idem XGBoost tunado — arquivado |
| `predicoes_walk_forward_ridge_baseline.csv` | **Essencial** | Predições do Ridge baseline |
| `predicoes_walk_forward_lightgbm.csv` | **Candidato à remoção** | Versão sem tuning — substituída pela versão tunada |
| `predicoes_walk_forward_random_forest.csv` | **Candidato à remoção** | Idem |
| `predicoes_walk_forward_xgboost.csv` | **Candidato à remoção** | Idem |

#### Métricas

| Arquivo | Classificação | Motivo |
|---|---|---|
| `metricas_walk_forward_lightgbm_tuned.csv` | **Essencial** | Métricas por fold — LightGBM tunado |
| `metricas_walk_forward_randomforest_tuned.csv` | **Essencial** | Idem Random Forest tunado |
| `metricas_walk_forward_xgboost_tuned.csv` | **Importante** | Idem XGBoost tunado |
| `metricas_ridge_baseline.csv` | **Essencial** | Métricas do Ridge baseline |
| `metricas_walk_forward_lightgbm.csv` | **Candidato à remoção** | Versão sem tuning — substituída |
| `metricas_walk_forward_random_forest.csv` | **Candidato à remoção** | Idem |
| `metricas_walk_forward_xgboost.csv` | **Candidato à remoção** | Idem |
| `tabela_metricas_tunadas_3modelos.csv` | **Candidato à remoção** | Versão sem Ridge — substituída pela tabela de 4 modelos |
| `tabela_metricas_tunadas_3modelos_resumo.csv` | **Candidato à remoção** | Idem |
| `tabela_metricas_preliminares_3modelos.csv` | **Candidato à remoção** | Versão pré-tuning — substituída pelas tunadas |
| `tabela_metricas_preliminares_3modelos_resumo.csv` | **Candidato à remoção** | Idem |

#### Relatórios de sessão (rastreabilidade cronológica)

Gerados dia a dia durante a Semana 2. Têm valor de rastreabilidade histórica mas não são referenciados por nenhum script.

| Arquivo | Classificação | Motivo |
|---|---|---|
| `relatorio_segunda_semana2_xgboost.txt` | **Temporário** | Relatório da segunda-feira da S2 — XGBoost |
| `relatorio_segunda_semana2_lightgbm.txt` | **Temporário** | Idem — LightGBM |
| `relatorio_terca_semana2_modelos_preliminares.txt` | **Temporário** | Relatório da terça |
| `relatorio_terca_semana2_random_forest.txt` | **Temporário** | Idem — Random Forest |
| `relatorio_quarta_semana2_xgboost_tuning.txt` | **Temporário** | Relatório da quarta — tuning XGBoost |
| `relatorio_quinta_semana2_randomforest_tuning.txt` | **Temporário** | Relatório da quinta — tuning RF |
| `relatorio_quinta_semana2_lightgbm_tuning.txt` | **Temporário** | Relatório da quinta — tuning LightGBM |
| `relatorio_quinta_semana2_ridge_baseline.txt` | **Temporário** | Relatório da quinta — Ridge |
| `relatorio_modelos_tunados_26_28_05.txt` | **Importante** | Consolidação dos 4 modelos tunados — inclui tabela comparativa final |
| `relatorio_feature_importance_29_30_05.txt` | **Essencial** | Relatório de importância de features — inclui top-10 de cada modelo e fold 2024 |
| `validacao_schema_2025_modelagem.txt` | **Essencial** | Resultado da validação de schema OpenF1 2025 vs. dataset de modelagem |
