# 11 — Plano de Estudos de Ablação

## Objetivo

Este documento define um plano de estudos de ablação para aproximar os modelos LightGBM e XGBoost das métricas-alvo do projeto. O objetivo não é apenas melhorar uma métrica isolada, mas avaliar mudanças no pipeline com base no equilíbrio entre:

- MAE;
- RMSE;
- R²;
- Kendall τ;
- top-3 accuracy.

O baseline considerado é o estado atual do pipeline após:

- RFE temporal multi-fold com 13 features;
- `decay=0.99`;
- tuning multi-métrica;
- finalistas atuais: LightGBM e XGBoost.

---

## Estado atual

| Métrica | Meta | LightGBM atual | XGBoost atual | Status |
|---|---:|---:|---:|---|
| MAE | ≤ 2.5 | 2.326 | 2.348 | Meta atingida |
| RMSE | ≤ 3.0 | 3.015 | 3.021 | Parcialmente atingida |
| R² | ≥ 0.75 | 0.658 | 0.657 | Meta não atingida |
| Kendall τ | ≥ 0.60 | 0.653 | 0.652 | Meta atingida |
| Top-3 accuracy | ≥ 70% | 25.6% | 25.6% | Meta não atingida |

Leitura principal:

- MAE e Kendall τ já estão dentro da meta.
- RMSE está próximo da meta, mas ainda acima de 3.0.
- R² tem gap relevante em relação a 0.75.
- Top-3 accuracy tem gap muito grande e provavelmente exige um modelo específico de pódio, não apenas regressão de posição final.

---

## Critério de decisão

Uma ablação só deve ser incorporada ao pipeline oficial se:

1. melhorar o score composto médio; ou
2. melhorar uma métrica crítica sem degradação relevante das demais; ou
3. resolver uma fragilidade metodológica documentável.

Critério recomendado:

| Métrica normalizada | Peso |
|---|---:|
| MAE invertido | 0.30 |
| RMSE invertido | 0.15 |
| R² | 0.20 |
| Kendall τ | 0.20 |
| Top-3 accuracy | 0.15 |

Também devem ser registradas melhorias específicas em RMSE e R², porque são as metas contínuas ainda mais próximas de serem melhoradas.

---

## Fase 1 — Congelar baseline

Antes de executar novas ablações, congelar os artefatos atuais:

- features finais: 13;
- decay: `0.99`;
- folds: 2023, 2024 e 2025;
- modelos foco: LightGBM e XGBoost;
- score atual:
  - LightGBM: `0.4971`;
  - XGBoost: `0.4963`.

Todo experimento deve comparar contra esse baseline.

---

## Fase 2 — Ablação de features

Objetivo: verificar se features de baixa importância estão adicionando ruído.

| Experimento | Hipótese |
|---|---|
| Remover `season_factor` | Pode estar pouco informativa em árvores, especialmente com decay alto |
| Remover `tire_compound_start` | Baixa importância nos dois modelos |
| Remover `grid_penalty` | Baixa importância e possível ruído |
| Remover `altitude_m` | Baixa importância isolada |
| Remover `avg_pit_stops_circuit` | Entrou no RFE multi-fold, mas tem importância baixa |
| Remover ranks 9-13 um a um | Ver se RMSE/R² melhoram |
| Usar top-10 features | Reduzir ruído |
| Usar top-8 features | Testar modelo mais enxuto |
| Reintroduzir `incident_rate_hist_norm` | Pode ajudar top-3/ranking em corridas caóticas |
| Reintroduzir `driver_dnf_rate` | Pode ajudar top-3 ou erros extremos |
| Reintroduzir `recent_form_3` apenas como controle | Confirmar redundância com `recent_form_5` |

Prioridade: alta.

---

## Fase 3 — Ablação de decay

O `decay=0.99` venceu a busca fina pelo score composto, mas deve ser comparado contra alternativas próximas com retuning completo dos modelos.

| Decay | Motivo |
|---|---|
| 0.95 | Melhor RMSE final anterior do LightGBM |
| 0.96 | Meio-termo |
| 0.97 | Meio-termo |
| 0.98 | Meio-termo |
| 0.99 | Atual vencedor |
| 1.00 | Sem decay; testar se o dataset favorece histórico quase integral |

Importante: cada decay precisa ser avaliado com novo tuning de LightGBM e XGBoost. Comparar decay sem retuning pode levar a conclusão incorreta.

Prioridade: alta.

---

## Fase 4 — Ablação dos pesos do score composto

Como RMSE e R² ainda estão abaixo da meta, testar pesos alternativos.

| Perfil | MAE | RMSE | R² | Kendall | Top-3 | Objetivo |
|---|---:|---:|---:|---:|---:|---|
| Atual | 0.30 | 0.15 | 0.20 | 0.20 | 0.15 | Equilíbrio geral |
| RMSE/R² | 0.25 | 0.25 | 0.25 | 0.15 | 0.10 | Aproximar RMSE ≤ 3 e R² |
| Ranking | 0.20 | 0.10 | 0.15 | 0.25 | 0.30 | Melhorar ranking e pódio |
| Erro contínuo | 0.40 | 0.30 | 0.20 | 0.05 | 0.05 | Melhorar MAE/RMSE |
| Pódio | 0.15 | 0.10 | 0.10 | 0.25 | 0.40 | Ver teto de top-3 |

Prioridade: alta.

---

## Fase 5 — Ablação de função objetivo

Objetivo: reduzir erros grandes, principalmente para melhorar RMSE.

LightGBM:

- `regression`;
- `regression_l1`;
- `huber`;
- `fair`.

XGBoost:

- `reg:squarederror`;
- `reg:absoluteerror`;
- `reg:pseudohubererror`.

Hipótese:

- Huber pode reduzir erros extremos sem sacrificar tanto MAE.
- Square error pode ajudar RMSE, mas pode piorar ranking.

Prioridade: média-alta.

---

## Fase 6 — Ablação de target

O target atual é `finish_position`. Como `qualifying_position` domina a importância, testar targets que modelem ganho/perda de posição pode melhorar R² e RMSE.

| Target | Hipótese |
|---|---|
| `finish_position` | Baseline atual |
| `log1p(finish_position)` | Reduz peso de posições ruins |
| `finish_position_rank_norm` por corrida | Normaliza tamanho e distribuição por GP |
| `delta_grid_to_finish` | Modelo prevê ganho/perda em relação à largada |
| Modelo em dois estágios: delta + recomposição | Pode melhorar R² e top-3 |

Prioridade: alta, mas exige cuidado metodológico e documentação.

---

## Fase 7 — Modelo específico para top-3

A meta de top-3 ≥ 70% é difícil para regressão de posição final. Para aproximar essa métrica, testar um problema separado de pódio.

| Experimento | Modelo |
|---|---|
| Regressão atual → top-3 por menor posição prevista | Baseline |
| Classificador binário `is_podium` | LightGBMClassifier / XGBClassifier |
| Learning-to-rank por corrida | XGBoost ranker |
| Ensemble regressão + classificador de pódio | Score híbrido |
| Calibração de probabilidade de pódio | Top-3 por probabilidade |

Recomendação: manter a regressão para MAE, RMSE, R² e Kendall τ, e tratar top-3 como estudo complementar de classificação.

Prioridade: muito alta se a banca cobrar a meta de top-3.

---

## Fase 8 — Ensembles

Ridge tem melhor MAE, RMSE, R² e Kendall τ, enquanto árvores têm melhor top-3. Portanto, testar combinações pode melhorar o equilíbrio geral.

| Ensemble | Hipótese |
|---|---|
| 70% Ridge + 30% LightGBM | Reduz RMSE mantendo não-linearidade |
| 50% Ridge + 50% LightGBM | Equilíbrio |
| 70% LightGBM + 30% XGBoost | Estabilidade entre árvores |
| Média LightGBM/XGBoost/RF | Pode melhorar top-3 |
| Stacking temporal simples | Pode melhorar score, mas exige cuidado com leakage |

Prioridade: alta.

---

## Ordem recomendada

1. Ablação simples de features: ranks baixos, top-10 e top-8.
2. Decay fino: `0.95`, `0.97`, `0.99`, `1.00`, sempre com retuning.
3. Pesos alternativos do score composto.
4. Funções objetivo: Huber, L1 e square error.
5. Target `delta_grid_to_finish`.
6. Ensemble Ridge + LightGBM/XGBoost.
7. Classificador específico de pódio.

---

## Resultado esperado

Expectativa realista:

| Métrica | Faixa esperada após ablações |
|---|---:|
| MAE | 2.25–2.35 |
| RMSE | tentar reduzir para < 3.0 |
| R² | 0.68–0.70 parece plausível; 0.75 é difícil sem novas features |
| Kendall τ | manter > 0.65 |
| Top-3 por regressão | 25–35% |
| Top-3 com classificador dedicado | potencialmente maior, mas como estudo separado |

Resumo: para aproximar as metas, o caminho mais promissor é atacar RMSE/R² com feature ablation, target delta e ensemble com Ridge; e atacar top-3 com um classificador dedicado de pódio.

---

## Execução inicial do plano

**Script executado:** `src/estudos_ablacao_modelos.py`

**Artefatos gerados:**

- `reports/ablacao/resultados_estudos_ablacao.csv`
- `reports/ablacao/relatorio_estudos_ablacao.md`

Esta primeira rodada avaliou ablações usando os hiperparâmetros atuais dos modelos, sem retuning completo para cada cenário. Portanto, os resultados abaixo indicam candidatos promissores, mas ainda não substituem a decisão oficial do pipeline.

### Melhores candidatos por score composto

| Experimento | Modelo | MAE | RMSE | R² | Kendall τ | Top-3 | Score |
|---|---|---:|---:|---:|---:|---:|---:|
| `target_delta_grid_to_finish` | LightGBM | **2.2833** | 3.0474 | 0.6498 | 0.6472 | **29.9%** | **0.5030** |
| `media_arvores` | Ensemble | 2.3400 | 3.0165 | 0.6580 | 0.6515 | 29.8% | 0.5028 |
| `decay_0.95` sem retuning | LightGBM | 2.3227 | 3.0131 | 0.6586 | 0.6500 | 28.4% | 0.5011 |
| `lgb_objective_regression_l1` | LightGBM | 2.2929 | 3.0277 | 0.6554 | 0.6493 | 27.0% | 0.4993 |
| `decay_0.97` sem retuning | XGBoost | 2.3499 | 3.0220 | 0.6565 | **0.6535** | 27.0% | 0.4984 |

### Melhores candidatos por RMSE

| Experimento | Modelo | RMSE | Observação |
|---|---|---:|---|
| `ridge_70_xgb_30` | Ensemble | **2.9583** | Melhor RMSE, mas top-3 cai para 21.5% |
| `ridge_70_lgb_30` | Ensemble | 2.9586 | Melhor equilíbrio entre erro contínuo e Kendall |
| `ridge_50_lgb_50` | Ensemble | 2.9671 | Ainda abaixo da meta RMSE ≤ 3.0 |
| `remove_avg_pit_stops_circuit` | LightGBM | 3.0107 | Melhora RMSE, mas piora top-3 |
| `decay_1.00` sem retuning | LightGBM | 3.0114 | Melhora RMSE, mas piora top-3 |

### Melhores candidatos por top-3

| Experimento | Modelo | Top-3 | Observação |
|---|---|---:|---|
| `target_delta_grid_to_finish` | LightGBM | **29.9%** | Melhor top-3 válido com regressão |
| `media_arvores` | Ensemble | 29.8% | Quase empate com target delta |
| `decay_0.95` sem retuning | LightGBM | 28.4% | Precisa retuning para comparação justa |
| `lgb_objective_regression_l1` | LightGBM | 27.0% | Melhora MAE e top-3 |
| `XGBClassifier` para pódio | Classificador | 27.0% | Não superou regressão/ensemble nesta rodada |

### Achados principais

1. **Target delta é o candidato mais promissor.**  
   Modelar `delta_grid_to_finish` melhorou MAE e top-3 de forma clara. O custo foi piora em RMSE e R². Deve ser retunado antes de qualquer decisão oficial.

2. **Ensemble das árvores quase empatou com target delta.**  
   A média LightGBM/XGBoost/Random Forest elevou top-3 para 29.8% e manteve R² próximo do baseline. É uma alternativa simples e forte.

3. **Ensembles com Ridge resolvem RMSE, mas sacrificam top-3.**  
   `ridge_70_xgb_30` e `ridge_70_lgb_30` ficam abaixo de RMSE 3.0 e sobem R² para ~0.671, mas derrubam top-3 para 21.5%.

4. **Feature ablation isolada não trouxe ganho dominante.**  
   Remover `avg_pit_stops_circuit` melhora RMSE em LightGBM/XGBoost, mas geralmente piora top-3. Reintroduzir `incident_rate_hist_norm` quase empata com baseline no LightGBM.

5. **Classificador de pódio simples não resolveu a meta top-3.**  
   O XGBClassifier chegou a 27.0%, abaixo de target delta e ensemble. Um estudo de pódio ainda pode valer, mas precisa tuning específico e talvez features próprias.

### Próximos experimentos recomendados

1. Retunar LightGBM e XGBoost usando `target_delta_grid_to_finish`.
2. Retunar LightGBM com `objective=regression_l1`.
3. Testar ensemble `media_arvores` e variantes ponderadas após retuning dos modelos candidatos.
4. Testar `decay=0.95`, `0.97`, `0.99` e `1.00` com retuning completo, porque a rodada sem retuning favoreceu `0.95` para top-3.
5. Criar estudo específico de ensemble multiobjetivo:
   - perfil RMSE/R²: maior peso para Ridge;
   - perfil top-3: maior peso para árvores;
   - perfil geral: otimizar pesos por score composto em folds 2023-2024.

Conclusão da execução inicial: o caminho mais promissor para melhorar o score composto é `target_delta_grid_to_finish` ou ensemble das árvores. Para bater RMSE ≤ 3.0, os ensembles com Ridge são os melhores, mas exigem aceitar queda de top-3.

---

## Retuning do candidato `target_delta_grid_to_finish`

**Script executado:** `src/tuning_target_delta_ablacao.py`

**Artefatos gerados:**

- `reports/ablacao/resultados_target_delta_retuned.csv`
- `reports/ablacao/resultado_lightgbm_target_delta_retuned.csv`
- `reports/ablacao/resultado_xgboost_target_delta_retuned.csv`
- `reports/ablacao/metricas_lightgbm_target_delta_retuned.csv`
- `reports/ablacao/metricas_xgboost_target_delta_retuned.csv`
- `reports/ablacao/params_lightgbm_target_delta_retuned.json`
- `reports/ablacao/params_xgboost_target_delta_retuned.json`

Como `target_delta_grid_to_finish` foi o melhor candidato da triagem inicial, ele foi retunado com Optuna para LightGBM e XGBoost. Esta etapa é mais justa do que comparar apenas com os hiperparâmetros herdados do alvo original.

### Resultado após retuning

| Experimento | Modelo | MAE | RMSE | R² | Kendall τ | Top-3 | Score |
|---|---|---:|---:|---:|---:|---:|---:|
| `XGBoost_target_delta_retuned` | XGBoost | **2.2657** | **3.0033** | **0.6601** | **0.6575** | 25.8% | **0.4997** |
| `LightGBM_target_delta_retuned` | LightGBM | 2.2800 | 3.0224 | 0.6562 | 0.6513 | 25.8% | 0.4981 |
| baseline oficial | LightGBM | 2.3264 | 3.0146 | 0.6582 | 0.6530 | 25.6% | 0.4971 |
| baseline oficial | XGBoost | 2.3479 | 3.0207 | 0.6569 | 0.6523 | 25.6% | 0.4963 |

### Interpretação

O melhor ganho confirmado foi o `XGBoost_target_delta_retuned`. Ele melhora o score composto em relação aos dois modelos oficiais e melhora principalmente MAE, RMSE, R² e Kendall τ. O RMSE fica muito próximo da meta de 3.0, mas ainda ligeiramente acima.

O ganho de top-3 observado na triagem inicial do LightGBM com `target_delta_grid_to_finish` (29.9%) não se sustentou após retuning: os dois modelos retunados ficaram em 25.8%, praticamente empatados com o baseline oficial. Portanto, `target_delta_grid_to_finish` é um bom caminho para erro contínuo e ordenação global, mas não resolve sozinho a meta de top-3.

### Decisão técnica após a execução

Para uma decisão orientada ao melhor resultado geral nas cinco métricas, o candidato mais forte da ablação é `XGBoost_target_delta_retuned`. Ele ainda não deve substituir automaticamente o pipeline oficial sem uma etapa final de confirmação, porque muda a definição operacional do alvo modelado: o modelo passa a prever o deslocamento entre grid e chegada, e a posição final é reconstruída a partir do grid.

Os próximos passos recomendados são:

1. validar `XGBoost_target_delta_retuned` como candidato oficial nos documentos 09 e 10;
2. testar ensemble entre `XGBoost_target_delta_retuned`, LightGBM oficial e Ridge para tentar reduzir RMSE abaixo de 3.0 sem derrubar top-3;
3. manter estudo separado para top-3, pois regressão e target delta ainda não ultrapassaram 30% de forma robusta após retuning.

---

## Execução completa do plano de ablação

**Scripts executados:**

- `src/estudos_ablacao_completo.py`
- `src/consolidar_estudos_ablacao_completo.py`

**Artefatos gerados:**

- `reports/ablacao/resultados_estudos_ablacao_completo.csv`
- `reports/ablacao/relatorio_estudos_ablacao_completo.md`
- `reports/ablacao/predicoes_ensemble_otimizado_melhor.csv`
- `reports/ablacao/metricas_*.csv`
- `reports/ablacao/predicoes_*.csv`
- `reports/ablacao/params_*.json`

A execução completa ampliou a rodada anterior e cobriu os eixos principais do plano:

- ablação de features;
- decay fino com retuning;
- perfis alternativos de score composto;
- funções objetivo alternativas;
- targets alternativos;
- classificadores específicos de pódio;
- ensembles otimizados.

Para os ensembles, foi usada uma grade discreta restrita, com passo 0.1, cobrindo pares e trios estratégicos. A grade exaustiva combinando todos os modelos simultaneamente foi descartada operacionalmente por explosão combinatória, sem mudar o objetivo do estudo: avaliar se combinações Ridge/árvores/target-delta melhoram o equilíbrio geral das métricas.

### Melhores resultados após execução completa

| Experimento | Modelo | MAE | RMSE | R² | Kendall τ | Top-3 | Score | Decisão |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `target_rank_norm_retuned` | LightGBM | **2.2381** | **2.8912** | **0.6857** | 0.6396 | **28.4%** | **0.5063** | Melhor resultado exploratório; requer cautela por usar escala derivada de `y_valid` |
| `score_erro_continuo_retuned` | LightGBM | 2.3113 | 2.9974 | 0.6621 | 0.6524 | **28.4%** | 0.5022 | Boa alternativa, mas inferior ao rank norm |
| `target_rank_norm_grid20_retuned` | LightGBM | 2.3227 | 3.0109 | 0.6589 | 0.6532 | 28.3% | 0.5013 | Melhor `rank_norm` causal por score |
| `ensemble_grid_rank_14` | Ensemble | 2.2517 | 2.9613 | 0.6700 | 0.6569 | 25.6% | 0.5013 | Melhor ensemble por score |
| `target_rank_norm_grid20_retuned` | XGBoost | 2.3063 | 2.9959 | 0.6623 | 0.6542 | 27.0% | 0.5005 | Melhor `rank_norm` causal para RMSE |
| `target_rank_norm_retuned` | XGBoost | 2.2542 | 2.8992 | 0.6839 | 0.6433 | 24.1% | 0.4996 | Bom RMSE/R², mas top-3 menor |
| `XGBoost_target_delta_retuned` | XGBoost | 2.2657 | 3.0033 | 0.6601 | **0.6575** | 25.8% | 0.4997 | Superado pelo rank norm |
| baseline oficial | LightGBM | 2.3264 | 3.0146 | 0.6582 | 0.6530 | 25.6% | 0.4971 | Referência anterior |

### Interpretação

O melhor candidato exploratório após a execução completa é `target_rank_norm_retuned` com LightGBM. Ele supera o baseline oficial em score composto, MAE, RMSE, R² e top-3 accuracy. A única queda relevante é em Kendall τ, que passa de 0.6530 para 0.6396, mas permanece acima da meta de 0.60.

Após auditoria metodológica, porém, esse resultado não deve substituir diretamente o pipeline oficial, porque a reconstrução da posição final usava o tamanho efetivo da corrida calculado em `y_valid`. Como a base exclui DNFs/DSQs, esse tamanho pode refletir informação pós-corrida. Para validar a ideia sem esse risco, foi criado o modo `rank_norm_grid20`, que normaliza e reconstrói a posição usando escala fixa de 20 posições, conhecida antes da corrida.

Em relação às metas do projeto, esse candidato:

- atinge MAE ≤ 2.5;
- atinge RMSE ≤ 3.0;
- mantém Kendall τ ≥ 0.60;
- melhora R² para 0.6857, mas ainda não atinge 0.75;
- melhora top-3 para 28.4%, mas ainda fica distante da meta de 70%.

O maior top-3 bruto apareceu em `xgb_objective_reg_pseudohubererror_retuned`, com 29.9%, mas esse resultado deve ser rejeitado porque o modelo ficou degenerado: MAE 22.5747, RMSE 23.1578 e R² -19.1253. Portanto, entre os modelos válidos, o melhor top-3 robusto ficou em 28.4%.

### Decisão técnica após a execução completa

Para uma escolha orientada ao melhor resultado geral nas cinco métricas sem risco de leakage, a melhor alternativa validada passa a ser `target_rank_norm_grid20_retuned`. O LightGBM tem maior score composto (`0.5013`) e top-3 (`28.3%`), enquanto o XGBoost tem melhor RMSE (`2.9959`) e R² (`0.6623`) entre as duas versões causais.

O `XGBoost_target_delta_retuned` continua útil como candidato secundário e controle metodológico, porque melhora MAE e Kendall τ em relação ao baseline, mas não é o melhor resultado final da ablação completa.

O resultado também confirma que mexer apenas em decay, pesos do score, objetivo de perda ou ensembles não foi suficiente para superar o ganho obtido pela mudança de target para rank normalizado por corrida. No entanto, a versão causal tem ganho menor que a versão exploratória. Portanto, a decisão mais segura é manter o pipeline original como oficial por enquanto e documentar `target_rank_norm_grid20_retuned` como candidato causal para possível substituição, caso a arquitetura seja atualizada para aceitar target normalizado por corrida.

### Validação causal do `rank_norm`

**Script executado:** `src/validar_rank_norm_causal.py`

**Artefatos gerados:**

- `reports/ablacao/resultados_rank_norm_causal.csv`
- `reports/ablacao/predicoes_target_rank_norm_grid20_retuned_lightgbm.csv`
- `reports/ablacao/predicoes_target_rank_norm_grid20_retuned_xgboost.csv`
- `reports/ablacao/metricas_target_rank_norm_grid20_retuned_lightgbm.csv`
- `reports/ablacao/metricas_target_rank_norm_grid20_retuned_xgboost.csv`
- `reports/ablacao/params_target_rank_norm_grid20_retuned_lightgbm.json`
- `reports/ablacao/params_target_rank_norm_grid20_retuned_xgboost.json`

| Experimento causal | Modelo | MAE | RMSE | R² | Kendall τ | Top-3 | Score | Leitura |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `target_rank_norm_grid20_retuned` | LightGBM | 2.3227 | 3.0109 | 0.6589 | 0.6532 | **28.3%** | **0.5013** | Melhor score causal |
| `target_rank_norm_grid20_retuned` | XGBoost | **2.3063** | **2.9959** | **0.6623** | **0.6542** | 27.0% | 0.5005 | Melhor erro/R² causal |
| baseline oficial | LightGBM | 2.3264 | 3.0146 | 0.6582 | 0.6530 | 25.6% | 0.4971 | Referência oficial |

Essa validação confirma que a ideia de target normalizado ainda melhora o score composto e o top-3 mesmo sem usar escala pós-corrida. Porém, o ganho é menor do que no `rank_norm` exploratório. Por isso, a substituição do pipeline original só deve ocorrer se o texto metodológico for atualizado para assumir explicitamente que o problema passa a prever posição final a partir de um target intermediário normalizado em escala fixa de grid.
