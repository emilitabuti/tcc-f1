# Inventário — docs/

Classificações: **Essencial** | **Importante** | **Temporário** | **Candidato à remoção**

---

## Documentos de referência do projeto

| Arquivo | Classificação | Motivo |
|---|---|---|
| `ArquiteturaProposta.pdf` | **Essencial** | Documento-mãe da arquitetura técnica. Define fontes, features, algoritmos, métricas e referências bibliográficas. Toda decisão do projeto deve ser rastreável até aqui |
| `Cronograma_Revisado_LightGBM.pdf` | **Essencial** | Cronograma de execução vigente. Inclui a decisão de adicionar LightGBM como terceiro algoritmo e o detalhamento dia a dia da Semana 1 e 2 |
| `Cronograma_Revisado.pdf` | **Importante** | Versão anterior ao LightGBM — mantida para mostrar a evolução da decisão |

---

## Guia e auditoria (criados nesta sessão)

| Arquivo | Classificação | Motivo |
|---|---|---|
| `GUIA_AUDITORIA.md` | **Essencial** | Guia de execução desta auditoria — define etapas, entregáveis e ordem de execução |
| `inventario/inventario_dados.md` | **Essencial** | Inventário de `data/raw/` e `data/processed/` |
| `inventario/inventario_scripts.md` | **Essencial** | Inventário de `src/` |
| `inventario/inventario_modelos.md` | **Essencial** | Inventário de `models/` e `reports/` |
| `inventario/inventario_documentacao.md` | **Essencial** | Este arquivo |

---

## Documentação metodológica existente

Gerada ao longo das Semanas 1 e 2 pelos próprios scripts do pipeline.

| Arquivo | Classificação | O que documenta | Observação |
|---|---|---|---|
| `metodologia_tratamento_dnf.md` | **Essencial** | Critérios de classificação de DNF e exclusão da base de modelagem | Gerada por `tratamento_dnf.py` |
| `metodologia_encoding.md` | **Essencial** | Estratégias de encoding por tipo de variável (OHE, ordinal, RAPM) | Gerada por `encoding.py` |
| `metodologia_normalizacao.md` | **Essencial** | Decisões de normalização: z-score vs. MinMax vs. sem normalização | Gerada por `normalizacao.py` |
| `metodologia_tratamento_valores_ausentes.md` | **Essencial** | Estratégia por tipo de variável (KNN, mediana, moda) | Gerada automaticamente |
| `metodologia_tratamento_outliers.md` | **Essencial** | Critério 3σ por circuito, distinção outlier legítimo vs. espúrio | Gerada por `tratamento_outliers.py` |
| `metodologia_preparacao_feature_engineering.md` | **Essencial** | Preparação da base FE-ready: track_complexity, weather histórico, pit stops | Gerada por `09_preparar_base_feature_engineering.py` |
| `metodologia_rapm_ridge.md` | **Essencial** | RAPM com Ridge e time-decay: matriz esparsa, causalidade, alpha, decay | Gerada por `rapm_ridge.py` |
| `metodologia_feature_engineering.md` | **Essencial** | Features históricas causais: recent_form, experience, wins, DNF rates, sinergia | Gerada por `feature_engineering_parte_1.py` |

---

## Análises e mapeamentos

| Arquivo | Classificação | O que documenta |
|---|---|---|
| `analise_feature_engineering_25_05_2026.md` | **Essencial** | Auditoria de leakage de 25/05 — identifica `safety_car_flag` e `weather_impact_factor` como leakage e prescreve correções. Evento crítico da transição S1→S2 |
| `lista_features_modelo.md` | **Essencial** | Contrato das 13 features finais do modelo — deve permanecer sincronizado com `models/feature_selection/features_modelagem_2018_2025.json` |
| `mapeamento_openf1_ergast.md` | **Essencial** | Mapeamento de identificadores entre OpenF1 e Ergast (driver_number → driver_id, meeting_key → round) |

---

## Documentação técnica criada (Etapa 3 desta auditoria)

| Arquivo | Etapa | O que documenta |
|---|---|---|
| `tecnico/01_coleta_dados.md` | 3.1 | Documenta decisão de usar 3 fontes, corte em 2018, RaceID |
| `tecnico/02_limpeza_dnf.md` | 3.2 | Documenta classificação de DNF, viés de sobrevivência, desclassificados |
| `tecnico/03_encoding_normalizacao.md` | 3.3 | Documenta por que OHE, ordinal, RAPM — e por que não normalizar para árvores |
| `tecnico/04_valores_ausentes_outliers.md` | 3.4 | Documenta KNN, critério 3σ, safety_car_flag como outlier legítimo |
| `tecnico/05_rapm_ridge.md` | 3.5 | Documenta RAPM como adaptação (não estrito), alpha=10.0 não tunado, decay=0.75 |
| `tecnico/06_feature_engineering.md` | 3.6 | Documenta cada feature com fórmula, referência e mecanismo causal |
| `tecnico/07_selecao_features.md` | 3.7 | Documenta leakages corrigidos, multicolinearidade e RFE |
| `tecnico/08_walk_forward_time_decay.md` | 3.8 | Documenta folds, decay=0.99 vs. 0.75 do paper, impossibilidade de embaralhar |
| `tecnico/09_modelagem_tuning.md` | 3.9 | Documenta 4 modelos, Optuna/grid search, decisão LightGBM+Random Forest e Ridge como baseline forte |
| `tecnico/10_resultados_feature_importance.md` | 3.10 | Documenta resultados, Ridge vs. árvores, feature importance e convergência com literatura |
| `tecnico/11_plano_estudos_ablacao.md` | Complementar | Documenta plano e resultado de ablações com target oficial fixo |
| `tecnico/12_baselines_literatura.md` | Complementar | Consolida baselines acadêmicos comparáveis e remoção de top-3 |
| `tecnico/13_material_complementar_discussao_modelos.md` | Complementar | Consolida discussão sobre literatura, linearidade aparente e mudanças regulatórias |
