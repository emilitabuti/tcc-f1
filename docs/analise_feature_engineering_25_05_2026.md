# ANÁLISE COMPLETA DO TCC — F1 Predictive Model
## Validação da Etapa de Feature Engineering até 23/05/2026

**Data da análise:** 25/05/2026  
**Escopo:** Validação de tudo executado até 23/05, com foco na etapa de Feature Engineering

> **Atualização pós-correção — 25/05/2026; revisada em 09/06/2026:** os bloqueadores identificados nesta análise foram tratados no pipeline. O dataset final de modelagem agora tem **13 features**, sem `safety_car_flag`, sem `grid_position`, sem `recent_form_3` e sem clima real observado da corrida. A RFE temporal multi-fold com XGBoost selecionou o subconjunto final por score composto multi-métrica. Artefatos principais: `data/processed/dataset_modelagem_X_2018_2025.csv`, `models/feature_selection/features_modelagem_2018_2025.json`, `models/feature_selection/relatorio_rfe_xgboost.txt`.

---

## 1. DIAGNÓSTICO GERAL

O pipeline de dados e feature engineering está **estruturalmente correto** em sua concepção e implementação técnica. A causalidade temporal foi cuidadosamente preservada nos cálculos históricos, o anti-leakage das colunas pós-corrida evidentes (finish_position, points, fastest_lap_race) está bem documentado, e o RAPM Ridge com iteração corrida-a-corrida é metodologicamente sólido.

No entanto, a análise identifica **2 problemas de Data Leakage que comprometem diretamente a validade científica** do modelo preditivo e **3 problemas de multicolinearidade severa** que não foram resolvidos, além de discrepâncias relevantes entre arquitetura proposta e implementação. O projeto **não pode avançar para modelagem sem tratar os dois problemas de leakage**.

---

## 2. VALIDAÇÃO DO QUE FOI ENTREGUE ATÉ 21/05

### Status de entrega por dia

| Dia | Tarefa | Status |
|---|---|---|
| Segunda 17/05 | Configuração ambiente + extração Ergast + FastF1 + OpenF1 | ✅ Concluído |
| Terça 18/05 | Limpeza Ergast/FastF1 + RaceID + DNF + Encoding + Normalização | ✅ Concluído |
| Quarta 19/05 | Valores ausentes + outliers + dataset limpo | ✅ Concluído |
| Quinta 20/05 | RAPM Ridge com time-decay (rapm_ridge.py) | ✅ Concluído |
| Sexta 21/05 | FE Parte 2: driver_experience, driver_wins_total, driver_dnf_rate, constructor_dnf_rate, constructor_wins_total, driver_constructor_synergy | ✅ Concluído |
| Fim de semana 22-23/05 | FE Parte 3 (track_complexity, weather_impact_factor, avg_pit_stops_circuit) + correlação + OpenF1 2025 | ✅ Concluído (além do previsto para Sexta) |

O cronograma da Semana 1 foi integralmente executado. Os entregáveis internos previstos para 23/05 estão presentes no repositório.

### Entregáveis verificados

- [x] `dataset_feature_engineering_ready_2018_2025.csv` — existe
- [x] `dataset_features_final_2018_2025.csv` — existe
- [x] `dataset_features_final_2018_2025_sem_nan.csv` — existe
- [x] `dataset_modelagem_X_2018_2025.csv` (2943 linhas, 21 colunas) — existe
- [x] `dataset_modelagem_y_2018_2025.csv` — existe
- [x] `coef_pilotos_rapm_2018_2025.csv` — existe
- [x] `coef_construtores_rapm_2018_2025.csv` — existe
- [x] `rapm_ridge.py` — implementado
- [x] `feature_engineering_parte_1.py` — implementado
- [x] `selecao_features_modelagem.py` — implementado
- [x] `analise_correlacao_features.py` — implementado
- [x] `openf1_2025_clean.csv` — existe
- [x] `mapeamento_openf1_ergast.md` — existe
- [x] `relatorio_correlacao.md` — existe
- [x] `manifest_feature_engineering.json` — existe

---

## 3. PROBLEMAS CRÍTICOS — DATA LEAKAGE

### 3.1 LEAKAGE CONFIRMADO: `safety_car_flag`

**Severidade: BLOQUEADOR**

A feature `safety_car_flag` é um flag binário por corrida (0/1) que indica se AQUELA corrida específica teve Safety Car ou Virtual Safety Car, derivado do `FastF1 TrackStatus 4/6/7` coletado durante a sessão.

**Evidência no código** (`src/09_preparar_base_feature_engineering.py`):

```python
mask_sc = (
    (df["outlier_flag"] == 1)
    & (df["outlier_tipo"] == "outlier_revisao")
    & (df["safety_car_flag"] == 1)
)
```

A flag é usada para reclassificar outliers ocorridos em corridas com SC — confirmando que é um dado per-corrida real. O manifesto também confirma: `"safety_car_corridas_2018_2025": 126`.

**O problema:** Um modelo preditivo de F1 prediz o resultado ANTES da corrida. Você não sabe se haverá Safety Car antes que ela aconteça. Usar `safety_car_flag = 1` como feature de entrada significa que durante a validação walk-forward o modelo "vê" informação que só existe após a corrida.

**O que é válido:** O componente causal `incident_rate_hist` (taxa histórica de SC por circuito, calculado com `expanding().mean().shift(1)`) em `track_complexity` está correto — é uma probabilidade histórica estimada antes da corrida.

**Impacto:** `safety_car_flag` tem correlação r=-0.085 com o target. Embora fraca, inclui informação post-race.

**Solução obrigatória:** Substituir `safety_car_flag` em X pelo `incident_rate_hist_norm` já calculado (taxa histórica de SC no circuito). Essa coluna já existe no dataset — é só adicioná-la à lista de features finais em substituição ao flag.

---

### 3.2 LEAKAGE CONFIRMADO: `weather_impact_factor`

**Severidade: BLOQUEADOR**

A fórmula de `weather_impact_factor`, conforme documento de arquitetura:

```
(humidity_norm + 2×rain + (1−air_temp_norm)) / 4
```

Os dados de temperatura, umidade e precipitação vêm da FastF1, que retorna dados de **telemetria coletados durante a sessão de corrida**. Isso inclui `rain` (chuva durante a corrida) e temperatura/umidade médias durante a corrida.

**Por que é leakage:** Se chove durante a corrida, você só sabe isso enquanto ela acontece. O modelo aprende: "quando chove, piloto X sobe posições" — mas ao predizer uma corrida futura, você não tem este valor. Você teria uma previsão meteorológica, não a chuva real.

**Sinal de alerta:** Correlação com target de r=-0.013 (essencialmente zero). Chuva na F1 é um dos maiores fatores de aleatoriedade — uma correlação próxima a zero pode indicar construção incorreta da feature ou que o sinal real está sendo diluído.

**Solução:** Recalcular `weather_impact_factor` usando dados climáticos históricos agregados por circuito e época do ano (ex: probabilidade histórica de chuva em Silverstone em julho, usando os dados FastF1 de temporadas anteriores). Caso contrário, remover e documentar como limitação.

---

## 4. PROBLEMAS DE MULTICOLINEARIDADE SEVERA NÃO RESOLVIDOS

### 4.1 `recent_form_5` × `recent_form_3`: r = 0.9874

**Severidade: Alta**

Correlação de 0.987 é funcionalmente redundância quase perfeita. Ambas são médias ponderadas da posição de chegada histórica — a diferença é apenas a janela (5 vs 3 corridas). A variação explicada por uma que não é explicada pela outra é inferior a 3%.

A justificativa "capturam janelas diferentes" é conceitualmente válida mas empiricamente irrelevante neste dataset.

**Decisão necessária:** Manter `recent_form_5` (janela maior, referência do RF+SHAP paper), remover `recent_form_3`. Documentar na metodologia.

**Detalhe crítico:** O arquivo `pares_correlacao_alta_maior_085.csv` tem `remover_sugerido = ""` para todos os 3 pares identificados — **zero features foram efetivamente removidas** após a análise de correlação, contrariando o processo previsto na arquitetura.

---

### 4.2 `grid_position` × `qualifying_position`: r = 0.9616

**Severidade: Alta**

A arquitetura original prevê apenas `grid_position`. A implementação adicionou `qualifying_position` e `grid_penalty` — expansão documentada e com justificativa válida. Porém, com r=0.962, as duas features de posição são quase idênticas para a esmagadora maioria das corridas.

**Dados observados:**
- `qualifying_position`: r=0.772 com target (maior de todas as features)
- `grid_position`: r=0.753 com target
- `grid_penalty`: r=-0.077 com target (fraco)

**Ação recomendada:** Remover `grid_position`, manter `qualifying_position` + `grid_penalty`. `qualifying_position` tem correlação maior e representa o desempenho puro no qualifying. `grid_position` é derivável dos dois e é redundante.

---

### 4.3 `recent_form_5` × `driver_constructor_synergy`: r = -0.8743

**Severidade: Moderada (próxima ao limiar de 0.85)**

`recent_form_5` usa `finish_position` e `driver_constructor_synergy` usa `-finish_position`. Ambas medem desempenho histórico recente do piloto, 87% coincidentes. A diferença conceitual (forma geral vs. forma com equipe específica) é legítima, mas precisa ser validada via SHAP.

**Ação:** Investigar via SHAP se `driver_constructor_synergy` adiciona importância além de `recent_form_5`. Se não, remover.

---

## 5. ANÁLISE INDIVIDUAL DAS 21 FEATURES

### 5.1 Grupo Grid (3 features)

| Feature | r com target | Avaliação |
|---|---|---|
| `qualifying_position` | 0.772 | ✅ Excelente — melhor feature do modelo. Manter. |
| `grid_position` | 0.753 | ⚠️ Redundante com qualifying_position (r=0.962). Remover. |
| `grid_penalty` | -0.077 | ⚠️ Fraca. Avaliar via SHAP. Candidata ao RFE. |

**`qualifying_position`** não estava na arquitetura original — adição correta e bem documentada.

---

### 5.2 Grupo Forma Recente (2 features)

| Feature | r com target | Avaliação |
|---|---|---|
| `recent_form_5` | 0.710 | ✅ Excelente. Implementação causal correta. Manter. |
| `recent_form_3` | 0.695 | ❌ Redundante com recent_form_5 (r=0.987). Remover. |

**Implementação causal:** `finish_position` histórico com shift(1) implícito — correto. Pesos (5,4,3,2,1) alinhados com RF+SHAP paper. Cold-start com 0.0 justificável.

---

### 5.3 Grupo Piloto (4 features)

| Feature | r com target | Avaliação |
|---|---|---|
| `driver_coef_rapm` | -0.599 | ✅ Boa. RAPM causal correto. Alpha=10 não tunado. |
| `driver_wins_total` | -0.350 | ✅ Moderada. Acumulado causal correto. |
| `driver_experience` | -0.207 | ⚠️ Fraca. Confundida por qualidade da equipe. Avaliar via RFE. |
| `driver_dnf_rate` | 0.083 | ⚠️ Muito fraca. Efeito diluído pelo DNF Excluded. |

**Problema metodológico no RAPM:** Alpha=10 não foi tunado via grid-search como previsto na arquitetura. O RAPM implementado é mais próximo de um Fixed Effects Model do que RAPM estrito — deve ser documentado como adaptação.

**Problema `driver_dnf_rate`:** O dataset usa DNF Excluded — as corridas onde o piloto DNF são removidas. Portanto, `driver_dnf_rate` histórica tem correlação quase nula com posição de chegada nas corridas que o piloto completa. Pode ser mais útil para predição de DNF do que de posição final.

---

### 5.4 Grupo Construtor (3 features)

| Feature | r com target | Avaliação |
|---|---|---|
| `constructor_coef_rapm` | -0.683 | ✅ Excelente. Maior correlação entre features de construtor. |
| `constructor_wins_total` | -0.421 | ✅ Moderada. Legado histórico capturado. |
| `constructor_dnf_rate` | -0.028 | ⚠️ Essencialmente zero. Mesmo problema do driver_dnf_rate. |

---

### 5.5 Sinergia (1 feature)

| Feature | r com target | Avaliação |
|---|---|---|
| `driver_constructor_synergy` | -0.663 | ⚠️ Forte, mas r=-0.874 com recent_form_5. Avaliar via SHAP. |

---

### 5.6 Grupo Circuito (3 features)

| Feature | r com target | Avaliação |
|---|---|---|
| `track_complexity` | -0.012 | ⚠️ Muito fraca. Pesos arbitrários. Avaliar via SHAP/interações. |
| `altitude_m` | -0.006 | ⚠️ Essencialmente zero. Efeito capturado indiretamente pelo RAPM. |
| `circuit_type` | -0.006 | ⚠️ Essencialmente zero. Candidata à remoção pelo RFE. |

**Nota `track_complexity`:** Os pesos (0.35 corners + 0.25 length + 0.20 altitude + 0.10 type + 0.10 incident_rate) são arbitrários sem calibração empírica. O componente `incident_rate_hist_norm` está correto causalmente, mas a correlação final com o target é próxima a zero.

---

### 5.7 Grupo Pneu (2 features)

| Feature | r com target | Avaliação |
|---|---|---|
| `tire_compound_start` | -0.001 | ⚠️ Literalmente zero. Efeito mascarado por estratégia e SC. |
| `avg_pit_stops_circuit` | -0.003 | ⚠️ Essencialmente zero. Prediz número de paradas, não quem vence. |

---

### 5.8 Grupo Temporada (1 feature)

| Feature | r com target | Avaliação |
|---|---|---|
| `season_factor` | 0.035 | ⚠️ Essencialmente zero. Provavelmente redundante com RAPM. |

**Problema adicional:** Para walk-forward com treino 2018-2024 e validação 2025, `season_factor=2025` é valor fora do range de treino. Tree-based models não extrapolam, mas a dependência temporal já está capturada pelo RAPM.

---

### 5.9 Clima e Eventos

| Feature | r com target | Avaliação |
|---|---|---|
| `weather_impact_factor` | -0.013 | ❌ LEAKAGE — dado during-race. Ver seção 3.2. |
| `safety_car_flag` | -0.085 | ❌ LEAKAGE — dado post-race. Ver seção 3.1. |

---

## 6. RELAÇÃO COM REFERÊNCIAS BIBLIOGRÁFICAS

| Referência | Features derivadas | Alinhamento |
|---|---|---|
| Henderson et al. (RAPM paper) | `driver_coef_rapm`, `constructor_coef_rapm` | ✅ Alinhado. Time-decay=0.75 correto. Alpha=10 não tunado. |
| Ruan et al. (RF+SHAP paper) | `recent_form_5`, `driver_dnf_rate`, `driver_constructor_synergy`, `track_complexity`, `weather_impact_factor` | ⚠️ Parcial. `recent_form_5` correto. `weather_impact_factor` com leakage. |
| Barra et al. (Advanced ML paper) | `grid_position`, `qualifying_position` | ✅ Alinhado. Correlação observada (0.753) > esperada (0.71). |
| Snoeks (RAPM adaptação) | `driver_coef_rapm`, `constructor_coef_rapm` | ✅ Alinhado |
| Heilmeier et al. | `avg_pit_stops_circuit` | ⚠️ Correlação quase zero — contribuição para posição final não comprovada |
| Tan et al. (causalidade) | Anti-leakage geral nas features históricas | ✅ Bem implementado |
| Koopman | `qualifying_position` com fallback para `grid_position` | ✅ Documentado corretamente |

### Features sem embasamento bibliográfico explícito
- `grid_penalty` — adicionada sem citação bibliográfica específica
- `season_factor` — razoável conceitualmente, mas sem citação de trabalho que usou o ano como feature direta

---

## 7. RANKING DE RISCO DAS FEATURES

| Feature | Problema | Risco | Ação |
|---|---|---|---|
| `safety_car_flag` | Leakage direto (dado pós-corrida) | **CRÍTICO** | Substituir por `incident_rate_hist_norm` |
| `weather_impact_factor` | Leakage (weather real da corrida) | **CRÍTICO** | Recalcular histórico ou remover |
| `recent_form_3` | Redundante com `recent_form_5` (r=0.987) | **Alto** | Remover |
| `grid_position` | Redundante com `qualifying_position` (r=0.962) | **Alto** | Remover |
| `tire_compound_start` | r=-0.001 | Moderado | Avaliar via SHAP antes de manter |
| `avg_pit_stops_circuit` | r=-0.003 | Moderado | Avaliar via SHAP |
| `circuit_type` | r=-0.006 | Moderado | Avaliar via SHAP/interações |
| `altitude_m` | r=-0.006 | Moderado | Avaliar via SHAP |
| `track_complexity` | r=-0.012, pesos arbitrários | Moderado | Avaliar via SHAP |
| `season_factor` | r=0.035, redundante com RAPM | Baixo | Avaliar necessidade |
| `constructor_dnf_rate` | r=-0.028 | Baixo | Candidata ao RFE |
| `driver_dnf_rate` | r=0.083 | Baixo | Candidata ao RFE |
| `grid_penalty` | r=-0.077 | Baixo | Candidata ao RFE |

---

## 8. PROBLEMAS METODOLÓGICOS IDENTIFICADOS

### 8.1 RFE com XGBoost nunca foi executado

A arquitetura propõe: ~20 features → análise de correlação (remove r > 0.85) → **RFE com XGBoost → 12-15 features finais**.

A implementação pulou o RFE e fixou 21 features manualmente em `selecao_features_modelagem.py`, sem treinar qualquer modelo. O `FEATURES_FINAIS` está hard-coded. O dataset foi congelado como se o RFE já tivesse sido executado.

**Risco:** A redução para 12-15 features (prevista para 29-30/05) pode alterar o conjunto final. O contrato de features não deveria ser "congelado" antes do RFE.

### 8.2 Análise de correlação sem decisão efetiva de remoção

Os 3 pares com r > 0.85 têm `remover_sugerido = ""` no CSV de saída. Nenhuma feature foi removida. A função `sugerir_decisao()` só recomenda remoção para versões zscore/minmax — para pares de features finais genuínos, retorna "Revisar manualmente". A revisão manual nunca aconteceu.

### 8.3 Normalização e risco walk-forward

O manifesto documenta corretamente: "Em walk-forward, scalers devem ser ajustados somente no treino de cada fold." Mas os CSVs já têm colunas `_zscore` e `_minmax` pré-calculadas sobre toda a base 2018-2025 — potencial erro se essas colunas forem usadas no Ridge baseline sem refit por fold.

### 8.4 RAPM sem controle de posição de largada

O RAPM puro modela presença/ausência. Na F1, `grid_position` é um confundidor enorme: pilotos com carros melhores saem na frente E terminam na frente. Sem incluir `grid_position` como covariável no RAPM, os coeficientes de construtor podem estar contaminados pelo efeito de largada. O RAPM implementado é mais próximo de Fixed Effects Model — deve ser documentado.

### 8.5 Data range 2018 vs. 2014 (benchmarks podem divergir)

A arquitetura original dizia "2014+". A documentação (etapa 09) corretamente atualiza para 2018, justificado pela disponibilidade do FastF1. Mas os papers de benchmark (DNN lap time, RAPM) usam 2014+ — comparações diretas de MAE devem considerar que o conjunto de treino é 4 anos menor.

---

## 9. IMPACTO DOS PROBLEMAS NO RESTANTE DO TCC

| Problema | Impacto na Modelagem (S2) | Impacto na Banca | Impacto TrAdaBoost (Fase 2) |
|---|---|---|---|
| Leakage `safety_car_flag` | Métricas infladas artificialmente | **Invalidação científica** se descoberto | Leakage persiste |
| Leakage `weather_impact_factor` | Métricas marginalmente infladas | Idem | Idem |
| Multicolinearidade `recent_form_3/5` | Instabilidade no Ridge baseline | Questionamento metodológico | Propaga |
| `grid_position` + `qualifying_position` | Coeficientes instáveis em Ridge | Questionamento | Propaga |
| 8 features com correlação ≈ zero | Ruído, overfitting potencial | Questionamento sobre seleção | Propaga |
| RFE não executado | Features desnecessárias inflam espaço | Inconsistência com arquitetura | Modelo mais pesado |

---

## 10. PRIORIDADE DE CORREÇÃO

### Prioridade 1 — BLOQUEADORES (antes de qualquer treino de modelo)

1. **Remover `safety_car_flag` de X** — substituir por `incident_rate_hist_norm` (já existe no dataset como coluna auxiliar). Modificar `FEATURES_FINAIS` em `selecao_features_modelagem.py` e regenerar os datasets X.

2. **Resolver `weather_impact_factor`:**
   - **Opção A (preferível):** Recalcular como probabilidade histórica de chuva por circuito/mês usando dados FastF1 de temporadas anteriores. Usar `expanding().mean().shift(1)` por circuito + mês.
   - **Opção B:** Remover do modelo e documentar como limitação metodológica explícita.

### Prioridade 2 — Alta (resolver antes do RFE/tuning)

3. **Remover `recent_form_3`** — manter `recent_form_5`. Documentar na metodologia.

4. **Remover `grid_position`** — manter `qualifying_position` + `grid_penalty`.

5. **Atualizar `lista_features_modelo.md` e `selecao_features_modelagem.py`** com o conjunto corrigido (deve ficar com ~18 features antes do RFE).

6. **Regenerar datasets X** — `dataset_modelagem_X_2018_2025.csv` e `dataset_modelagem_X_2018_2024.csv`.

### Prioridade 3 — Média (resolver durante RFE na semana 2)

7. **Executar RFE com XGBoost** — com features corrigidas, rodar ranking e remover contribuição nula/mínima. Candidatas: `constructor_dnf_rate`, `driver_dnf_rate`, `circuit_type`, `altitude_m`, `avg_pit_stops_circuit`, `tire_compound_start`, `season_factor`, `grid_penalty`.

8. **Tunar alpha do RAPM Ridge** via grid-search na validação cruzada temporal.

9. **Investigar `driver_constructor_synergy` via SHAP** — se adicionar <5% de importância além de `recent_form_5`, remover.

---

## 11. CHECKLIST ANTES DE AVANÇAR PARA MODELAGEM

### Obrigatório (bloqueadores)

- [ ] `safety_car_flag` removida de X ou substituída por `incident_rate_hist_norm`
- [ ] `weather_impact_factor` recalculada como dado histórico pré-corrida ou removida
- [ ] Decisão documentada sobre `recent_form_3` (remover)
- [ ] Decisão documentada sobre `grid_position` (remover) vs `qualifying_position` (manter)
- [ ] `FEATURES_FINAIS` em `selecao_features_modelagem.py` atualizado
- [ ] `lista_features_modelo.md` atualizado
- [ ] `dataset_modelagem_X_2018_2025.csv` regenerado após correções
- [ ] `dataset_modelagem_X_2018_2024.csv` regenerado após correções

### Recomendado (antes do tuning)

- [ ] `pares_correlacao_alta_maior_085.csv` atualizado com decisões efetivas
- [ ] Verificar que normalizações z-score/minmax NÃO são usadas como features diretas — só para Ridge baseline com refit por fold
- [ ] Documentar adaptação RAPM vs. Fixed Effects Model na metodologia
- [ ] Verificar cobertura OpenF1 2025 para todos os GPs disputados até a data

---

## 12. VEREDICTO FINAL

### O projeto está pronto para avançar para modelagem? **SIM, após as correções de 25/05/2026**

O projeto está em estado avançado e a fundação é sólida. A implementação causal das features históricas é correta e cuidadosa. O pipeline é reprodutível e documentado. A estrutura de dados está limpa (zero NaN, zero RaceIDs duplicados).

Os **2 problemas de data leakage** foram removidos do conjunto final de modelagem:

- `safety_car_flag` permanece apenas como auditoria e foi substituída em X por `incident_rate_hist_norm`.
- `weather_impact_factor` foi recalculada como histórico causal por circuito, mas a RFE temporal não a manteve no subconjunto final.
- `recent_form_3` e `grid_position` foram removidas por redundância severa.
- A RFE com XGBoost reduziu o conjunto de candidatos corrigidos para **13 features finais** na revisão multi-fold.

Após as correções, o projeto está em condição adequada para iniciar a modelagem da Semana 2 com fundamento científico mais sólido. O único par com `|r| > 0.85` remanescente é `recent_form_5 × driver_constructor_synergy`; ele foi mantido por diferença conceitual e deve ser acompanhado via SHAP/ablação durante a modelagem.

---

## APÊNDICE — Correlações observadas com o target

| Feature | r com finish_position |
|---|---|
| qualifying_position | 0.7717 |
| grid_position | 0.7531 |
| recent_form_5 | 0.7104 |
| recent_form_3 | 0.6950 |
| constructor_coef_rapm | -0.6831 |
| driver_constructor_synergy | -0.6629 |
| driver_coef_rapm | -0.5992 |
| constructor_wins_total | -0.4206 |
| driver_wins_total | -0.3499 |
| driver_experience | -0.2072 |
| safety_car_flag | -0.0852 |
| driver_dnf_rate | 0.0829 |
| grid_penalty | -0.0768 |
| season_factor | 0.0351 |
| constructor_dnf_rate | -0.0278 |
| weather_impact_factor | -0.0128 |
| track_complexity | -0.0125 |
| altitude_m | -0.0058 |
| circuit_type | -0.0056 |
| avg_pit_stops_circuit | -0.0031 |
| tire_compound_start | -0.0014 |

## APÊNDICE — Pares com correlação > 0.85 (identificados, não resolvidos)

| Par | r | Decisão tomada | Ação necessária |
|---|---|---|---|
| recent_form_5 × recent_form_3 | 0.9874 | "Revisar manualmente" | **Remover recent_form_3** |
| grid_position × qualifying_position | 0.9616 | "Revisar manualmente" | **Remover grid_position** |
| recent_form_5 × driver_constructor_synergy | -0.8743 | "Revisar manualmente" | Avaliar via SHAP |
