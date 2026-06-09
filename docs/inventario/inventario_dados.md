# Inventário — data/

Classificações: **Essencial** | **Importante** | **Temporário** | **Candidato à remoção**

---

## data/raw/

Fontes originais extraídas das APIs. Nenhum arquivo aqui é gerado pelo pipeline — todos vieram de extração externa.

| Arquivo | Fonte | Classificação | Motivo |
|---|---|---|---|
| `ergast_2018_2024.csv` | Ergast API | **Essencial** | Base histórica principal: resultados, grid, DNF, pontos de 2018 a 2024 |
| `ergast_2025_results.csv` | Jolpica (Ergast migrado) | **Essencial** | Extensão 2025 dos resultados — alimenta o fold de validação final |
| `ergast_pitstop_2018_2025.csv` | Ergast API | **Essencial** | Dados de pit stop — origem de `avg_pit_stops_circuit` |
| `ergast_pitstop_2018_2025_parcial.csv` | Ergast API | **Candidato à remoção** | Versão parcial da extração, provavelmente sobreposta pelo arquivo completo acima. Verificar se conteúdo é subconjunto antes de remover |
| `fastf1_laps_2018_2025.csv` | FastF1 | **Essencial** | Tempos de volta por setor, compostos de pneu — origem de features de ritmo |
| `fastf1_qualifying_2018_2025.csv` | FastF1 | **Essencial** | Posições de qualifying — origem de `qualifying_position` (feature mais importante do modelo, r=0.77) |
| `fastf1_weather_2018_2025.csv` | FastF1 | **Essencial** | Dados climáticos — usados para calcular `weather_impact_factor` histórico |
| `fastf1_checkpoint_v2.json` | FastF1 (interno) | **Candidato à remoção** | Checkpoint de extração incremental. Só é útil para retomar uma extração interrompida. Não alimenta nenhuma etapa do pipeline |
| `jolpica_circuits.csv` | Jolpica | **Essencial** | Metadados de circuito: altitude, tipo (permanente/urbano), comprimento, curvas |
| `jolpica_drivers.csv` | Jolpica | **Essencial** | Metadados de piloto: nome, nacionalidade, data de nascimento |
| `circuitos_manual.csv` | Entrada manual | **Essencial** | Dados de circuito não disponíveis via API — preenchidos manualmente |
| `openf1_meetings_2025_2026.csv` | OpenF1 API | **Essencial** | Calendário 2025 e 2026 com meeting_keys — usado para mapear corridas e no script de atualização 2026 |
| `openf1_session_result_2025_2026.csv` | OpenF1 API | **Essencial** | Resultados das sessões de corrida 2025 e 2026 — alimenta o fold 2025 e a análise de drift 2026 |
| `openf1_starting_grid_2025.csv` | OpenF1 API | **Importante** | Grid de largada 2025 via endpoint `/starting_grid`. Atualmente tem 0 registros para 2026 — será complementado pelo `update_openf1_2026.py` via API em tempo de execução |
| `openf1_stints_2025_2026.csv` | OpenF1 API | **Importante** | Stints por piloto — origem do composto de largada (`tire_compound_start`) para 2026 |
| `openf1_race_control_2025_2026.csv` | OpenF1 API | **Importante** | Eventos de corrida (safety car, bandeiras) — origem do `safety_car_flag` |
| `openf1_weather_2025_2026.csv` | OpenF1 API | **Importante** | Dados climáticos 2025-2026 |
| `openf1_validation.json` | OpenF1 (interno) | **Candidato à remoção** | Artefato gerado durante a validação da extração. Não alimenta nenhuma etapa do pipeline |
| `dados_ausentes.txt` | Manual | **Importante** | Registro manual de lacunas identificadas na coleta original |

---

## data/processed/

Artefatos gerados pelo pipeline. Organizados por etapa de origem.

### Etapa 01 — Limpeza Ergast + FastF1

| Arquivo | Classificação | Motivo |
|---|---|---|
| `historico_ergast_fastf1_limpo_2018_2024.csv` | **Importante** | Versão 2018-2024 da base limpa — usada para comparações de fold |
| `historico_ergast_fastf1_limpo_2018_2025.csv` | **Essencial** | Base após limpeza inicial — ponto de entrada de todo o pipeline |
| `base_historica_limpa_2018_2024.csv` | **Candidato à remoção** | Parece duplicata de `historico_ergast_fastf1_limpo_2018_2024.csv`. Verificar se são idênticos |
| `base_historica_limpa_2018_2025.csv` | **Candidato à remoção** | Idem para 2018-2025. Verificar antes de remover |
| `relatorio_01_limpeza_ergast_fastf1_2018_2025.txt` | **Essencial** | Rastreabilidade da etapa 01: contagens, decisões de limpeza |

### Etapa 02 — Tratamento de DNF

| Arquivo | Classificação | Motivo |
|---|---|---|
| `historico_dnf_classificado_2018_2024.csv` | **Importante** | DNFs classificados (piloto/mecânico/outro) — versão 2018-2024 |
| `historico_dnf_classificado_2018_2025.csv` | **Essencial** | DNFs classificados — alimenta `driver_dnf_rate` e `constructor_dnf_rate` |
| `historico_dnf_excluded_2018_2024.csv` | **Importante** | Base sem DNFs — versão 2018-2024 para fold comparativo |
| `historico_dnf_excluded_2018_2025.csv` | **Essencial** | Base sem DNFs — alvo principal do modelo |
| `base_historica_dnf_classificado_2018_2024.csv` | **Candidato à remoção** | Verificar se é duplicata de `historico_dnf_classificado_2018_2024.csv` |
| `base_historica_dnf_classificado_2018_2025.csv` | **Candidato à remoção** | Idem para 2018-2025 |
| `base_historica_dnf_excluded_2018_2024.csv` | **Candidato à remoção** | Verificar duplicata |
| `base_historica_dnf_excluded_2018_2025.csv` | **Candidato à remoção** | Verificar duplicata |
| `relatorio_02_tratamento_dnf.txt` | **Essencial** | Rastreabilidade da etapa 02 |

### Etapa 03 — Encoding

| Arquivo | Classificação | Motivo |
|---|---|---|
| `historico_encoded_2018_2024.csv` | **Importante** | Base com encoding — versão 2018-2024 |
| `historico_encoded_2018_2025.csv` | **Importante** | Base com encoding aplicado |
| `base_historica_encoded_2018_2024.csv` | **Candidato à remoção** | Verificar duplicata de `historico_encoded_2018_2024.csv` |
| `base_historica_encoded_2018_2025.csv` | **Candidato à remoção** | Idem |
| `relatorio_03_encoding.txt` | **Essencial** | Rastreabilidade da etapa 03 |

### Etapa 04 — Normalização

| Arquivo | Classificação | Motivo |
|---|---|---|
| `historico_normalizado_2018_2024.csv` | **Importante** | Base normalizada — versão 2018-2024 |
| `historico_normalizado_2018_2025.csv` | **Importante** | Base com z-score e MinMax aplicados |
| `base_historica_normalizado_2018_2024.csv` | **Candidato à remoção** | Verificar duplicata |
| `base_historica_normalizado_2018_2025.csv` | **Candidato à remoção** | Verificar duplicata |
| `relatorio_04_normalizacao.txt` | **Essencial** | Rastreabilidade da etapa 04 |

### Etapa 05 — Valores Ausentes

| Arquivo | Classificação | Motivo |
|---|---|---|
| `relatorio_05_tratamento_valores_ausentes.txt` | **Essencial** | Rastreabilidade da etapa 05 |

### Etapa 06 — Outliers

| Arquivo | Classificação | Motivo |
|---|---|---|
| `historico_outliers_tratados_2018_2024.csv` | **Importante** | Base com outliers tratados — versão 2018-2024 |
| `historico_outliers_tratados_2018_2025.csv` | **Importante** | Base com outliers tratados |
| `historico_imputado_normalizado_2018_2024.csv` | **Candidato à remoção** | Estágio intermediário entre normalização e outliers. Regenerável pelo pipeline |
| `historico_imputado_normalizado_2018_2025.csv` | **Candidato à remoção** | Idem |
| `outliers_removidos_2018_2024.csv` | **Importante** | Registro dos outliers removidos — 2018-2024 |
| `outliers_removidos_2018_2025.csv` | **Importante** | Registro dos outliers removidos |
| `outliers_revisao_2018_2025.csv` | **Importante** | Outliers marcados para revisão manual |
| `relatorio_06_tratamento_outliers.txt` | **Essencial** | Rastreabilidade da etapa 06 |

### Etapa 07 — Integração de Fontes

| Arquivo | Classificação | Motivo |
|---|---|---|
| `relatorio_07_integracao_fontes.txt` | **Essencial** | Rastreabilidade da etapa 07 |

### Etapa 08 — Processamento OpenF1 2025

| Arquivo | Classificação | Motivo |
|---|---|---|
| `openf1_2025_clean.csv` | **Essencial** | Alias de `validacao_2025_clean.csv` — fold 2025 pronto (419 linhas, 24 GPs) |
| `validacao_2025_clean.csv` | **Essencial** | Fold 2025 do walk-forward — dataset principal de validação |
| `relatorio_08_openf1_2025.txt` | **Essencial** | Rastreabilidade da etapa 08 |

### Etapa 09 — Preparação para Feature Engineering

| Arquivo | Classificação | Motivo |
|---|---|---|
| `dataset_pre_features_2018_2024.csv` | **Temporário** | Estágio intermediário — entrada da etapa 09 para versão 2024 |
| `dataset_pre_features_2018_2025.csv` | **Temporário** | Idem para 2018-2025. Regenerável via pipeline |
| `dataset_feature_engineering_ready_2018_2024.csv` | **Importante** | Base FE-ready versão 2018-2024 — usada pelo RAPM para gerar coeficientes do fold 2024 |
| `dataset_feature_engineering_ready_2018_2025.csv` | **Essencial** | Entrada do RAPM Ridge e do Feature Engineering |
| `relatorio_09_preparacao_feature_engineering.txt` | **Essencial** | Rastreabilidade da etapa 09 |

### Etapa 10 — RAPM Ridge

| Arquivo | Classificação | Motivo |
|---|---|---|
| `coef_pilotos_rapm_2018_2025.csv` | **Essencial** | Coeficientes RAPM por piloto — origem de `driver_coef_rapm` |
| `coef_construtores_rapm_2018_2025.csv` | **Essencial** | Coeficientes RAPM por construtor — origem de `constructor_coef_rapm` |
| `coef_pilotos.csv` | **Candidato à remoção** | Cópia legada de `coef_pilotos_rapm_2018_2025.csv`. Verificar se ainda é referenciada por algum script antes de remover |
| `coef_construtores.csv` | **Candidato à remoção** | Idem |
| `relatorio_10_rapm_ridge.txt` | **Essencial** | Rastreabilidade da etapa 10 |

### Etapa 11 — Feature Engineering

| Arquivo | Classificação | Motivo |
|---|---|---|
| `dataset_features_final_2018_2024.csv` | **Importante** | Base com todas as features — versão 2018-2024, necessária para análise de drift |
| `dataset_features_final_2018_2025.csv` | **Essencial** | Base completa com todas as features geradas |
| `dataset_features_final_2018_2025_sem_nan.csv` | **Essencial** | Versão sem NaN — entrada direta do script de seleção de features |
| `dataset_feature_engineering_parte_1_2018_2024.csv` | **Temporário** | Estágio intermediário do FE versão 2024. Regenerável |
| `manifest_feature_engineering.json` | **Essencial** | Rastreabilidade completa de todas as transformações aplicadas |
| `relatorio_11_feature_engineering_parte_1.txt` | **Essencial** | Rastreabilidade da etapa 11 |
| `relatorio_feature_engineering.txt` | **Candidato à remoção** | Versão preliminar do relatório — sobreposta pelo `relatorio_11`. Verificar se tem conteúdo exclusivo |
| `relatorio_feature_engineering_final.txt` | **Candidato à remoção** | Idem |
| `relatorio_feature_engineering_2018_2024.txt` | **Importante** | Relatório da versão 2024 — manter para rastreabilidade do fold |

### Etapa 13 — Seleção de Features e Dataset de Modelagem

| Arquivo | Classificação | Motivo |
|---|---|---|
| `dataset_modelagem_2018_2025.csv` | **Importante** | Dataset unificado com X + y + chaves |
| `dataset_modelagem_X_2018_2024.csv` | **Essencial** | Matrix X versão 2018-2024 — referência para análise de drift |
| `dataset_modelagem_X_2018_2025.csv` | **Essencial** | Matrix X final de modelagem (13 features, 2.943 linhas) |
| `dataset_modelagem_y_2018_2024.csv` | **Essencial** | Target versão 2018-2024 |
| `dataset_modelagem_y_2018_2025.csv` | **Essencial** | Target final (finish_position, 2.943 linhas) |
| `target_finish_position_2018_2024.csv` | **Candidato à remoção** | Verificar se é duplicata de `dataset_modelagem_y_2018_2024.csv` |
| `target_finish_position_2018_2025.csv` | **Candidato à remoção** | Idem para 2018-2025 |

### Semana 3 — Dados 2026

| Arquivo | Classificação | Motivo |
|---|---|---|
| `openf1_2026_available.csv` | **Essencial** | 4 corridas de 2026 processadas — base para análise de drift; realinhar schema para as 13 features finais antes de nova rodada |
| `relatorio_update_2026.txt` | **Importante** | Rastreabilidade da extração e processamento 2026 |
