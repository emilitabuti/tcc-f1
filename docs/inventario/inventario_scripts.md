# Inventário — src/

Classificações: **Essencial** | **Importante** | **Temporário** | **Candidato à remoção**

Nota: a pasta `src/__pycache__/` contém apenas bytecode Python gerado automaticamente. Todos os arquivos `.pyc` ali são **candidatos à remoção** — são regenerados em tempo de execução e não devem estar no repositório.

---

## Coleta de dados

Scripts que extraem dados das APIs externas. Executados uma única vez (ou pontualmente para atualizações).

| Arquivo | Classificação | O que faz | Saída principal |
|---|---|---|---|
| `extract_ergast_results.py` | **Essencial** | Extrai resultados históricos da Ergast/Jolpica API | `data/raw/ergast_2018_2024.csv` |
| `extract_ergast_2025.py` | **Essencial** | Extrai resultados de 2025 da Jolpica | `data/raw/ergast_2025_results.csv` |
| `extract_ergast_pitstop.py` | **Essencial** | Extrai dados de pit stop | `data/raw/ergast_pitstop_2018_2025.csv` |
| `extract_fastf1.py` | **Essencial** | Extrai tempos de volta, qualifying e clima via FastF1 | `data/raw/fastf1_*.csv` |
| `extract_jolpica_circuits.py` | **Essencial** | Extrai metadados de circuitos | `data/raw/jolpica_circuits.csv` |
| `extract_jolpica_drivers.py` | **Essencial** | Extrai metadados de pilotos | `data/raw/jolpica_drivers.csv` |
| `extract_openf1_race_data.py` | **Essencial** | Extrai resultados, stints, race control e clima OpenF1 2025-2026 | `data/raw/openf1_*.csv` |
| `extract_openf1_starting_grid_2025.py` | **Importante** | Extrai grid de largada via endpoint `/starting_grid` da OpenF1 | `data/raw/openf1_starting_grid_2025.csv` |
| `connect_openf1.py` | **Importante** | Módulo utilitário de conexão à API OpenF1 — usado pelos scripts de extração |  — |

---

## Pipeline de dados (etapas sequenciais)

Scripts executados em ordem para transformar os dados raw em datasets prontos para modelagem.

| Arquivo | Etapa | Classificação | O que faz | Saída principal |
|---|---|---|---|---|
| `limpeza_ergast_fastf1.py` | 01 | **Essencial** | Filtra era híbrida (2018+), cria RaceID, remove nulos, une Ergast e FastF1 | `historico_ergast_fastf1_limpo_2018_2025.csv` |
| `tratamento_dnf.py` | 02 | **Essencial** | Classifica DNFs em piloto/mecânico/outro e gera base excluindo DNFs | `historico_dnf_*.csv` |
| `encoding.py` | 03 | **Essencial** | Aplica OHE em circuito/construtor, ordinal em pneu | `historico_encoded_*.csv` |
| `normalizacao.py` | 04 | **Essencial** | Z-score para contínuas, MinMax para posição e voltas | `historico_normalizado_*.csv` |
| `tratamento_valores_ausentes.py` | 05 | **Essencial** | Imputa qualifying ausente (KNN), mediana para contínuas, moda para categóricas | — |
| `tratamento_outliers.py` | 06 | **Essencial** | Aplica critério 3σ por circuito, mantém outliers legítimos com flag | `historico_outliers_tratados_*.csv` |
| `07_integrar_fontes_suporte.py` | 07 | **Essencial** | Integra Ergast, FastF1 e Jolpica em base unificada | `relatorio_07_integracao_fontes.txt` |
| `08_processar_openf1_2025.py` | 08 | **Essencial** | Isola temporada 2025 do dataset completo como fold de validação | `validacao_2025_clean.csv` |
| `09_preparar_base_feature_engineering.py` | 09 | **Essencial** | Prepara base com colunas de circuito, clima histórico e track_complexity | `dataset_feature_engineering_ready_*.csv` |
| `rapm_ridge.py` | 10 | **Essencial** | Gera coeficientes RAPM via Ridge com time-decay, corrida a corrida (causal) | `coef_pilotos_rapm_*.csv`, `coef_construtores_rapm_*.csv` |
| `feature_engineering_parte_1.py` | 11 | **Essencial** | Cria recent_form, driver_experience, wins, DNF rates, sinergia | `dataset_features_final_*.csv` |
| `analise_correlacao_features.py` | 12 | **Essencial** | Calcula matriz de correlação de Pearson entre as features candidatas/finais | `reports/correlacao_features/` |
| `rfe_xgboost_features.py` | 12b | **Essencial** | RFE temporal multi-fold com XGBoost — seleciona as 13 features finais por score composto | `models/feature_selection/` |
| `selecao_features_modelagem.py` | 13 | **Essencial** | Gera X e y finais com as 13 features, remove colunas proibidas | `dataset_modelagem_X_*.csv`, `dataset_modelagem_y_*.csv` |
| `pipeline_dados.py` | Orquestrador | **Essencial** | Executa as etapas do pipeline em sequência | — |

---

## Modelagem

Scripts de treinamento, tuning e avaliação dos modelos.

| Arquivo | Classificação | O que faz | Saída principal |
|---|---|---|---|
| `metricas.py` | **Essencial** | Calcula MAE, RMSE, R² e Kendall τ; top-3 não faz parte do pipeline oficial | — (módulo utilitário) |
| `tuning_utils.py` | **Essencial** | Funções comuns de tuning Optuna compartilhadas entre os 3 modelos | — (módulo utilitário) |
| `otimizacao_time_decay.py` | **Essencial** | Grid-search do fator de time-decay por score composto (0.50 a 0.99) | `reports/modelagem/time_decay_escolhido_xgboost.txt` |
| `walk_forward.py` | **Essencial** | Walk-forward validation do XGBoost (3 folds: 2023, 2024, 2025) | `reports/modelagem/metricas_walk_forward_xgboost.csv` |
| `walk_forward_random_forest.py` | **Essencial** | Idem para Random Forest | `reports/modelagem/metricas_walk_forward_random_forest.csv` |
| `walk_forward_lightgbm.py` | **Essencial** | Idem para LightGBM | `reports/modelagem/metricas_walk_forward_lightgbm.csv` |
| `tuning_xgboost.py` | **Essencial** | Tuning Optuna do XGBoost (50 trials) | `reports/modelagem/optuna_xgboost_best_params.json` |
| `tuning_randomforest.py` | **Essencial** | Tuning Optuna do Random Forest (50 trials) | `reports/modelagem/optuna_randomforest_best_params.json` |
| `tuning_lightgbm.py` | **Essencial** | Tuning Optuna do LightGBM (50 trials) | `reports/modelagem/optuna_lightgbm_best_params.json` |
| `otimizacao_ridge_lambda.py` | **Essencial** | RidgeCV para baseline linear — varre alpha 0.01 a 100 | `reports/modelagem/ridge_best_params.json` |
| `consolidar_metricas_tunadas.py` | **Essencial** | Agrega métricas dos 4 modelos tunados em tabela comparativa | `reports/modelagem/tabela_metricas_tunadas_4modelos.csv` |
| `consolidar_metricas_preliminares.py` | **Importante** | Idem para versão preliminar (pré-tuning) | `reports/modelagem/tabela_metricas_preliminares_3modelos.csv` |
| `gerar_feature_importance_modelos.py` | **Essencial** | Extrai importância de features dos 3 modelos de árvore + salva fold 2024 | `reports/modelagem/feature_importance_*.csv` |
| `estudos_ablacao_modelos.py` | **Importante** | Estudos de ablação oficiais com target fixo `finish_position`; transformações de target foram desativadas | `reports/ablacao/` |
| `estudos_ablacao_completo.py` | **Importante** | Retunings de ablação oficiais com target fixo `finish_position`; transformações de target foram desativadas | `reports/ablacao/` |
| `ablacao_pareada_lgbm_xgboost.py` | **Importante** | Ablação pareada LightGBM vs. XGBoost com target oficial fixo `finish_position` | `reports/ablacao/pareada_lgbm_xgboost/` |
| `tuning_target_delta_ablacao.py` | **Temporário / obsoleto** | Experimento histórico com target transformado; bloqueado para uso oficial | — |
| `validar_rank_norm_causal.py` | **Temporário / obsoleto** | Experimento histórico com target transformado; bloqueado para uso oficial | — |

---

## Validação e suporte

| Arquivo | Classificação | O que faz |
|---|---|---|
| `mapear_openf1_ergast.py` | **Importante** | Mapeia identificadores entre OpenF1 e Ergast (driver_number → driver_id) |
| `validar_schema_2025_modelagem.py` | **Essencial** | Verifica compatibilidade de schema entre OpenF1 2025 e o dataset de modelagem |
| `verificar_completude.py` | **Importante** | Cross-check FastF1 — verifica se todos os GPs foram extraídos |
| `10_eda_validacao_dataset_tratado.py` | **Importante** | EDA de validação do dataset tratado — gera gráficos e sumário |

---

## Semana 3

| Arquivo | Classificação | O que faz |
|---|---|---|
| `update_openf1_2026.py` | **Essencial** | Extrai e processa corridas 2026 disponíveis, aplica as features finais, salva `openf1_2026_available.csv` |
