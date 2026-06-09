# Plano de Documentação — TCC F1 Predictive Model

Este documento define quais arquivos técnicos serão criados na Etapa 3, em que ordem, com quais fontes e quais perguntas cada um deve responder.

A ordem segue o fluxo real de execução do projeto.

---

## Estrutura de cada documento técnico

Todo arquivo em `docs/tecnico/` segue esta estrutura:

```
## Contexto
## Fundamentação bibliográfica
## Implementação
## Resultados obtidos
## Avaliação crítica
## Convergência com a literatura
```

O documento só é considerado **concluído** quando todas as seis seções estiverem preenchidas com evidências do código ou dos dados — não com descrições genéricas.

---

## Documento 01 — Coleta de Dados

**Arquivo:** `docs/tecnico/01_coleta_dados.md`

**Fontes a consultar:**
- `src/extract_ergast_results.py`, `extract_ergast_2025.py`, `extract_ergast_pitstop.py`
- `src/extract_fastf1.py`
- `src/extract_openf1_race_data.py`, `extract_openf1_starting_grid_2025.py`
- `docs/ArquiteturaProposta.pdf` — seção 1 (Fontes de Dados)

**Perguntas que o documento deve responder:**
1. Por que o corte é em 2018 e não em 2014 como outros papers usam?
2. Por que três fontes? O que cada uma oferece que as outras não têm?
3. Como o `RaceID` é construído e por que essa chave garante unicidade?
4. Como os dados do Ergast e FastF1 são sincronizados (qual é a chave de join)?
5. O que acontece quando um GP está no Ergast mas não no FastF1 (ou vice-versa)?
6. Como o OpenF1 difere do Ergast/FastF1 em termos de schema?

**Critério de conclusão:** todas as perguntas respondidas com referência ao código ou ao documento de arquitetura.

---

## Documento 02 — Limpeza e Tratamento de DNF

**Arquivo:** `docs/tecnico/02_limpeza_dnf.md`

**Fontes a consultar:**
- `src/limpeza_ergast_fastf1.py`
- `src/tratamento_dnf.py`
- `data/processed/relatorio_01_limpeza_ergast_fastf1_2018_2025.txt`
- `data/processed/relatorio_02_tratamento_dnf.txt`
- `docs/metodologia_tratamento_dnf.md`
- `docs/ArquiteturaProposta.pdf` — seção 2 (Tratamento dos Dados)

**Perguntas que o documento deve responder:**
1. Quais critérios definem um registro inválido que é removido na limpeza inicial?
2. Como os DNFs são classificados em piloto, mecânico e outro? Quais palavras-chave são usadas?
3. Por que DNFs são excluídos do dataset de modelagem e não apenas marcados com flag?
4. O que acontece com pilotos desclassificados (status `Disqualified`)? Eles são tratados como DNF?
5. Qual é o viés de sobrevivência introduzido pela exclusão de DNFs e como afeta o modelo?
6. Quantos registros são removidos em cada etapa da limpeza?

**Critério de conclusão:** todas as perguntas respondidas com números do relatório e trechos do código.

---

## Documento 03 — Encoding e Normalização

**Arquivo:** `docs/tecnico/03_encoding_normalizacao.md`

**Fontes a consultar:**
- `src/encoding.py`
- `src/normalizacao.py`
- `data/processed/relatorio_03_encoding.txt`
- `data/processed/relatorio_04_normalizacao.txt`
- `docs/metodologia_encoding.md`
- `docs/metodologia_normalizacao.md`
- `docs/ArquiteturaProposta.pdf` — seção 2 (Encoding e Normalização)

**Perguntas que o documento deve responder:**
1. Por que circuito e construtor usam OHE e não Label Encoding?
2. Por que composto de pneu usa Label Encoding ordinal (Soft=1, Medium=2, Hard=3)?
3. Por que piloto não usa OHE nem Label, mas sim um coeficiente RAPM?
4. Por que XGBoost e Random Forest não precisam de normalização, mas o Ridge precisa?
5. O z-score e MinMax calculados sobre o dataset completo (2018-2025) introduzem leakage no Ridge baseline quando usado no walk-forward?
6. Como os scalers são serializados em `models/preprocessing/` e por que isso é necessário?

**Critério de conclusão:** todas as perguntas respondidas com referência ao código e à arquitetura.

---

## Documento 04 — Valores Ausentes e Outliers

**Arquivo:** `docs/tecnico/04_valores_ausentes_outliers.md`

**Fontes a consultar:**
- `src/tratamento_valores_ausentes.py`
- `src/tratamento_outliers.py`
- `data/processed/relatorio_05_tratamento_valores_ausentes.txt`
- `data/processed/relatorio_06_tratamento_outliers.txt`
- `docs/metodologia_tratamento_valores_ausentes.md`
- `docs/metodologia_tratamento_outliers.md`
- `docs/ArquiteturaProposta.pdf` — seção 2 (Valores Ausentes e Outliers)

**Perguntas que o documento deve responder:**
1. Quais variáveis têm valores ausentes e qual é a estratégia para cada tipo?
2. Por que qualifying usa KNN e não mediana?
3. O critério de 3σ por circuito é aplicado por circuito ou globalmente? Por que por circuito?
4. Como outliers legítimos (safety car, falhas mecânicas) são distinguidos de outliers espúrios?
5. O que acontece com o `safety_car_flag` nessa etapa? Como ele é criado e por que foi depois identificado como leakage?
6. Quantos registros foram removidos como outliers espúrios e quantos foram mantidos com flag?

**Critério de conclusão:** todas as perguntas respondidas com números dos relatórios.

---

## Documento 05 — RAPM Ridge

**Arquivo:** `docs/tecnico/05_rapm_ridge.md`

**Fontes a consultar:**
- `src/rapm_ridge.py`
- `data/processed/relatorio_10_rapm_ridge.txt`
- `models/rapm/manifest_rapm_ridge.json`
- `docs/metodologia_rapm_ridge.md`
- `docs/ArquiteturaProposta.pdf` — seção 2 (Encoding, item Piloto) e seção 3 (Feature Engineering)
- Referência [9]: Henderson et al. — paper original do RAPM aplicado à F1
- Referência [10]: Snoeks — decomposição piloto vs. construtor

**Perguntas que o documento deve responder:**
1. O que é RAPM e de onde vem o conceito (Basketball → F1)?
2. Como o modelo implementado difere do RAPM original? Por que essa adaptação é válida?
3. Como a causalidade é garantida (treina só em corridas anteriores a r)?
4. Como o time-decay é calculado? Por que decay=0.75 por temporada (referência Henderson et al.)?
5. Por que o target é `-finish_position` e não `finish_position`?
6. O alpha=10.0 foi tunado ou é um default? Qual é o impacto dessa escolha?
7. O que é cold-start e como é tratado para pilotos e construtores sem histórico?
8. Como os coeficientes são incorporados ao dataset principal via merge?

**Critério de conclusão:** todas as perguntas respondidas, incluindo a admissão explícita de que alpha não foi tunado como previsto na arquitetura.

---

## Documento 06 — Feature Engineering

**Arquivo:** `docs/tecnico/06_feature_engineering.md`

**Fontes a consultar:**
- `src/feature_engineering_parte_1.py`
- `src/09_preparar_base_feature_engineering.py`
- `data/processed/relatorio_11_feature_engineering_parte_1.txt`
- `data/processed/relatorio_09_preparacao_feature_engineering.txt`
- `docs/metodologia_feature_engineering.md`
- `docs/metodologia_preparacao_feature_engineering.md`
- `docs/ArquiteturaProposta.pdf` — seção 3 (Feature Engineering)
- Referências [2] Ruan et al., [6] Heilmeier et al., [9] Henderson et al.

**Perguntas que o documento deve responder:**

Para cada feature criada:
1. Qual é a fórmula exata de cálculo?
2. Qual referência bibliográfica a fundamenta?
3. Como a causalidade é garantida (qual mecanismo de shift/expanding)?
4. Qual é o valor de cold-start e por que esse valor?

Features a documentar individualmente:
- `recent_form_5` e `recent_form_3` (incluindo por que `recent_form_3` foi removida depois)
- `driver_coef_rapm` e `constructor_coef_rapm`
- `driver_experience` e `driver_wins_total`
- `constructor_wins_total`
- `driver_dnf_rate` e `constructor_dnf_rate`
- `driver_constructor_synergy`
- `track_complexity` (incluindo os pesos arbitrários)
- `weather_impact_factor` (incluindo a versão original com leakage e a corrigida)
- `avg_pit_stops_circuit`
- `season_factor`
- `incident_rate_hist_norm`
- `qualifying_position` e `grid_penalty` (adicionadas além da arquitetura original)

**Critério de conclusão:** tabela completa com fórmula, referência, mecanismo causal e cold-start para cada feature.

---

## Documento 07 — Seleção de Features

**Arquivo:** `docs/tecnico/07_selecao_features.md`

**Fontes a consultar:**
- `src/analise_correlacao_features.py`
- `src/rfe_xgboost_features.py`
- `src/selecao_features_modelagem.py`
- `docs/analise_feature_engineering_25_05_2026.md`
- `reports/correlacao_features/relatorio_correlacao.md`
- `reports/correlacao_features/pares_correlacao_alta_maior_085.csv`
- `models/feature_selection/rfe_xgboost_ranking.csv`
- `models/feature_selection/rfe_xgboost_subsets.csv`

**Perguntas que o documento deve responder:**
1. Quais dois leakages foram identificados em 25/05 e como cada um foi corrigido?
2. Quais pares de multicolinearidade severa foram encontrados e qual foi a decisão para cada um?
3. Como o RFE foi executado temporalmente (por que não usar cross-validation padrão)?
4. Por que 13 features e não 12 ou 15? O que o `rfe_xgboost_subsets.csv` mostra?
5. Por que `driver_wins_total` foi excluída pelo RFE se estava na arquitetura original?
6. O par `recent_form_5` × `driver_constructor_synergy` (r=0.87) foi mantido — com qual justificativa?
7. Quais features ficaram fora do modelo e por quê?

**Critério de conclusão:** decisão documentada para cada feature incluída e excluída, com evidência numérica.

---

## Documento 08 — Walk-Forward e Time-Decay

**Arquivo:** `docs/tecnico/08_walk_forward_time_decay.md`

**Fontes a consultar:**
- `src/walk_forward.py`, `walk_forward_lightgbm.py`, `walk_forward_random_forest.py`
- `src/otimizacao_time_decay.py`
- `reports/modelagem/otimizacao_time_decay_xgboost.csv`
- `reports/modelagem/time_decay_escolhido_xgboost.txt`
- `docs/ArquiteturaProposta.pdf` — seção 8 (Divisão Treino/Teste)
- Referências [9] Henderson et al., [18] Tan et al.

**Perguntas que o documento deve responder:**
1. Por que walk-forward e não cross-validation padrão em séries temporais?
2. Como os três folds são definidos e por que começam em 2018 e não em 2014?
3. Como o time-decay é implementado como `sample_weight` no treinamento?
4. O fator 0.95 foi escolhido por qual critério? O que o grid-search mostrou?
5. Por que 0.95 difere do 0.75 recomendado por Henderson et al.? Essa divergência é defensável?
6. Por que os dados nunca são embaralhados?
7. Como as métricas MAE, RMSE, R², Kendall τ e top-3 são calculadas por fold?

**Critério de conclusão:** todas as perguntas respondidas com os valores reais do `otimizacao_time_decay_xgboost.csv`.

---

## Documento 09 — Modelagem e Tuning

**Arquivo:** `docs/tecnico/09_modelagem_tuning.md`

**Fontes a consultar:**
- `src/tuning_xgboost.py`, `tuning_randomforest.py`, `tuning_lightgbm.py`, `otimizacao_ridge_lambda.py`
- `reports/modelagem/optuna_*_best_params.json`
- `reports/modelagem/tabela_metricas_tunadas_4modelos_resumo.csv`
- `reports/modelagem/decisao_algoritmos.md`
- `docs/ArquiteturaProposta.pdf` — seção 9 (Algoritmos) e seção 10 (Métricas)
- Referências [19] Chen & Guestrin (XGBoost), [20] Breiman (RF), [3] Barra et al. (LightGBM), [22] Bergstra & Bengio, [23] Akiba et al. (Optuna)

**Perguntas que o documento deve responder:**
1. Por que XGBoost, Random Forest e LightGBM? Quais são as diferenças filosóficas entre eles?
2. Por que Optuna com 50 trials? Como o espaço de busca de cada modelo foi definido?
3. Por que LightGBM foi adicionado ao cronograma original (que previa só XGBoost e RF)?
4. Quais hiperparâmetros ótimos foram encontrados para cada modelo?
5. Por que LightGBM e Random Forest foram escolhidos como finalistas e XGBoost foi arquivado?
6. Por que o Ridge baseline supera os modelos de árvore em MAE médio?
7. Por que o Ridge permanece como baseline e não como finalista principal?

**Critério de conclusão:** decisão dos finalistas documentada com os números exatos da tabela comparativa.

---

## Documento 10 — Resultados e Feature Importance

**Arquivo:** `docs/tecnico/10_resultados_feature_importance.md`

**Fontes a consultar:**
- `reports/modelagem/tabela_metricas_tunadas_4modelos.csv`
- `reports/modelagem/tabela_metricas_tunadas_4modelos_resumo.csv`
- `reports/modelagem/feature_importance_lgb.csv`
- `reports/modelagem/feature_importance_rf.csv`
- `reports/modelagem/feature_importance_xgb.csv`
- `reports/modelagem/feature_importance_2024.csv`
- `reports/modelagem/relatorio_feature_importance_29_30_05.txt`
- Referências [1] Polishchuk, [2] Ruan et al., [3] Barra et al., [9] Henderson et al.

**Perguntas que o documento deve responder:**
1. Quais são os resultados finais de cada modelo em cada fold (tabela completa)?
2. As metas da arquitetura foram atingidas? (MAE ≤ 2.5, RMSE ≤ 3.0, R² ≥ 0.75, Kendall τ ≥ 0.60)
3. Por que o top-3 accuracy ficou abaixo de 30% se Polishchuk reporta 78%?
4. `qualifying_position` domina a feature importance nos três modelos — isso é coerente com a literatura?
5. `constructor_coef_rapm` aparece consistentemente no top-3 — o que isso confirma da literatura (Snoeks [10])?
6. Features de circuito e pneu têm importância baixa — isso invalida sua inclusão no modelo?
7. Como a importância do fold 2024 (`feature_importance_2024.csv`) será usada na análise de drift da Semana 3?

**Critério de conclusão:** tabela de resultados preenchida com todos os números reais e todas as perguntas respondidas com referência bibliográfica.

---

## Resumo do plano

| # | Documento | Arquivo | Fontes principais |
|---|---|---|---|
| 3.1 | Coleta de Dados | `tecnico/01_coleta_dados.md` | Scripts de extração, Arquitetura seção 1 |
| 3.2 | Limpeza e DNF | `tecnico/02_limpeza_dnf.md` | `limpeza_ergast_fastf1.py`, `tratamento_dnf.py`, relatórios 01-02 |
| 3.3 | Encoding e Normalização | `tecnico/03_encoding_normalizacao.md` | `encoding.py`, `normalizacao.py`, relatórios 03-04 |
| 3.4 | Valores Ausentes e Outliers | `tecnico/04_valores_ausentes_outliers.md` | `tratamento_valores_ausentes.py`, `tratamento_outliers.py`, relatórios 05-06 |
| 3.5 | RAPM Ridge | `tecnico/05_rapm_ridge.md` | `rapm_ridge.py`, relatório 10, refs [9][10] |
| 3.6 | Feature Engineering | `tecnico/06_feature_engineering.md` | `feature_engineering_parte_1.py`, relatório 11, refs [2][6][9] |
| 3.7 | Seleção de Features | `tecnico/07_selecao_features.md` | `rfe_xgboost_features.py`, correlação, `analise_feature_engineering_25_05_2026.md` |
| 3.8 | Walk-Forward e Time-Decay | `tecnico/08_walk_forward_time_decay.md` | Scripts walk-forward, otimização decay, refs [9][18] |
| 3.9 | Modelagem e Tuning | `tecnico/09_modelagem_tuning.md` | Scripts de tuning, best_params, `decisao_algoritmos.md`, refs [19][20][3] |
| 3.10 | Resultados e Feature Importance | `tecnico/10_resultados_feature_importance.md` | Tabelas de métricas, feature importance, refs [1][2][9] |
