# Registro de Limpeza do Repositório

Data de execução: 02/06/2026

---

## O que foi feito

### REMOVIDO (deletado permanentemente)

| Arquivo | Motivo | Risco | Evidência |
|---|---|---|---|
| `src/__pycache__/*.pyc` (43 arquivos) | Bytecode Python gerado automaticamente. Não pertence ao controle de versão. | Nenhum — regenerado em qualquer execução | Padrão Python |
| `reports/eda_dataset_tratado/__pycache__/` | Idem | Nenhum | Idem |
| `data/raw/ergast_pitstop_2018_2025_parcial.csv` | **Idêntico** ao `ergast_pitstop_2018_2025.csv` (verificado por hash MD5 e comparação `DataFrame.equals()`). 5.941 linhas, 7 colunas, conteúdo byte-a-byte idêntico. | Nenhum — arquivo completo está presente | Verificação em Python: `p.equals(f) == True` |

---

### ARQUIVADO — `reports/modelagem/historico_semana2/`

Relatórios gerados dia a dia durante a Semana 2 de modelagem. Têm valor de rastreabilidade histórica mas não são referenciados por nenhum script nem necessários para reproduzir os resultados finais. A versão consolidada e definitiva está em `relatorio_modelos_tunados_26_28_05.txt`.

| Arquivo arquivado | Conteúdo |
|---|---|
| `relatorio_segunda_semana2_xgboost.txt` | Walk-forward XGBoost sem tuning — segunda-feira |
| `relatorio_segunda_semana2_lightgbm.txt` | Walk-forward LightGBM sem tuning |
| `relatorio_terca_semana2_modelos_preliminares.txt` | Métricas preliminares dos 3 modelos — terça |
| `relatorio_terca_semana2_random_forest.txt` | Walk-forward RF sem tuning |
| `relatorio_quarta_semana2_xgboost_tuning.txt` | Tuning XGBoost — quarta |
| `relatorio_quinta_semana2_randomforest_tuning.txt` | Tuning RF — quinta |
| `relatorio_quinta_semana2_lightgbm_tuning.txt` | Tuning LightGBM — quinta |
| `relatorio_quinta_semana2_ridge_baseline.txt` | Ridge baseline — quinta |

---

### ARQUIVADO — `reports/modelagem/historico_pre_tuning/`

Versões intermediárias de métricas e predições, produzidas antes do tuning Optuna completo ou sem o Ridge baseline. Substituídas pelas versões definitivas em `tabela_metricas_tunadas_4modelos.csv` e arquivos `*_tuned.csv`.

| Arquivo arquivado | Por que foi substituído |
|---|---|
| `tabela_metricas_preliminares_3modelos.csv` | Versão pré-tuning — substituída por `tabela_metricas_tunadas_4modelos.csv` |
| `tabela_metricas_preliminares_3modelos_resumo.csv` | Idem |
| `tabela_metricas_tunadas_3modelos.csv` | Versão sem Ridge — substituída pela de 4 modelos |
| `tabela_metricas_tunadas_3modelos_resumo.csv` | Idem |
| `decisao_preliminar_algoritmos.md` | Versão anterior da decisão — substituída por `decisao_algoritmos.md` |
| `predicoes_walk_forward_lightgbm.csv` | Predições pré-tuning — substituídas por `predicoes_walk_forward_lightgbm_tuned.csv` |
| `predicoes_walk_forward_random_forest.csv` | Idem para RF |
| `predicoes_walk_forward_xgboost.csv` | Idem para XGBoost |
| `metricas_walk_forward_lightgbm.csv` | Métricas pré-tuning — substituídas por `metricas_walk_forward_lightgbm_tuned.csv` |
| `metricas_walk_forward_random_forest.csv` | Idem para RF |
| `metricas_walk_forward_xgboost.csv` | Idem para XGBoost |

---

### ARQUIVADO — `reports/eda_dataset_tratado/arquivado/`

| Arquivo arquivado | Motivo |
|---|---|
| `ydata_profile_dataset_tratado.json` (3.0 MB) | O HTML correspondente (`ydata_profile_dataset_tratado.html`) contém as mesmas informações em formato legível. O JSON é um artefato de processamento interno do ydata-profiling sem uso direto. |

---

## O que foi analisado mas mantido — com justificativa

### `data/processed/base_historica_*` (10 arquivos)

**Mantidos.** Verificação mostrou que esses arquivos são **diferentes** dos `historico_*` correspondentes. Eles são entradas explícitas dos scripts do pipeline:

| Arquivo | Referenciado por | Por que é diferente |
|---|---|---|
| `base_historica_limpa_*.csv` | `src/tratamento_dnf.py` (INPUT) | Versão paralela da base limpa — produzida via caminho alternativo em `limpeza_ergast_fastf1.py` |
| `base_historica_dnf_excluded_*.csv` | `src/encoding.py` (INPUT) | Resultado do DNF Excluded sobre a base histórica limpa |
| `base_historica_encoded_*.csv` | `src/normalizacao.py` (INPUT) | Resultado do encoding sobre a base histórica |
| `base_historica_normalizado_*.csv` | Referenciada na doc de `normalizacao.py` | Resultado da normalização sobre a base histórica |
| `base_historica_dnf_classificado_*.csv` | Referência geral | Versão classificada — mantida para rastreabilidade |

Remover esses arquivos quebraria a reexecução do pipeline.

### `data/processed/coef_pilotos.csv` e `coef_construtores.csv`

**Mantidos.** Verificação mostrou que são **idênticos** a `coef_pilotos_rapm_2018_2025.csv` e `coef_construtores_rapm_2018_2025.csv`. Porém, `src/feature_engineering_parte_1.py` lê diretamente `coef_pilotos.csv` (linha 24-25). Remover sem atualizar o script quebraria a Feature Engineering.

**Recomendação futura:** atualizar `feature_engineering_parte_1.py` para ler os arquivos com nome canônico (`*_rapm_2018_2025.csv`) e então remover as cópias legadas.

### `data/processed/historico_imputado_normalizado_*.csv`

**Mantidos.** Referenciados por `src/tratamento_outliers.py` como INPUT (linhas 19-20). São o estágio intermediário entre a normalização e o tratamento de outliers — necessários para reexecução.

### `data/processed/dataset_pre_features_*.csv`

**Mantidos.** Referenciados por `src/09_preparar_base_feature_engineering.py` e `src/08_processar_openf1_2025.py` como INPUT. São a etapa imediatamente anterior à base FE-ready.

### `data/processed/target_finish_position_*.csv`

**Mantidos.** Verificação mostrou que têm estrutura diferente do `dataset_modelagem_y_*.csv` — incluem colunas adicionais (`race_name`, `circuit_id`). São referenciados por `09_preparar_base_feature_engineering.py` como OUTPUT. Mantidos para rastreabilidade.

### `data/processed/relatorio_feature_engineering*.txt` (versões antigas)

**Mantidos.** São pequenos (~KB) e fazem parte da rastreabilidade do pipeline. Não causam confusão porque o nome `relatorio_feature_engineering_final.txt` claramente indica superposição — futura limpeza pode removê-los após confirmar que nenhum script os referencia.

### `data/raw/fastf1_checkpoint_v2.json`

**Mantido.** Permite retomar extração FastF1 interrompida sem re-baixar todo o histórico. Não interfere com o pipeline de modelagem.

### `data/raw/openf1_validation.json`

**Mantido.** Artefato pequeno de validação da extração OpenF1. Não interfere com o pipeline.

---

## Estado após a limpeza

### `reports/modelagem/` — arquivos ativos

```
decisao_algoritmos.md               ← decisão final revisada LightGBM + XGBoost
feature_importance_2024.csv         ← referência para drift análise
feature_importance_lgb.csv
feature_importance_rf.csv
feature_importance_xgb.csv
metricas_ridge_baseline.csv
metricas_walk_forward_lightgbm_tuned.csv
metricas_walk_forward_randomforest_tuned.csv
metricas_walk_forward_xgboost_tuned.csv
optuna_*_best_params.json (3 modelos)
optuna_*_trials.csv (3 modelos)
otimizacao_time_decay_xgboost.csv
otimizacao_time_decay_xgboost_resumo.csv
predicoes_walk_forward_*_tuned.csv (3 + ridge)
relatorio_feature_importance_29_30_05.txt
relatorio_modelos_tunados_26_28_05.txt
ridge_alpha_grid.csv
ridge_best_params.json
tabela_metricas_tunadas_4modelos.csv    ← tabela definitiva
tabela_metricas_tunadas_4modelos_resumo.csv
time_decay_escolhido_xgboost.txt
validacao_schema_2025_modelagem.txt
historico_semana2/                   ← 8 relatórios arquivados
historico_pre_tuning/                ← 11 arquivos arquivados
```

### Resumo quantitativo

| Ação | Quantidade | Espaço recuperado |
|---|---|---|
| Removido (`.pyc` + parcial) | 45 arquivos | ~15 MB (estimado) |
| Arquivado em subpastas | 20 arquivos + 1 JSON (3MB) | ~3 MB em pasta ativa |
| Mantido com justificativa documentada | ~30 arquivos candidatos | — |
