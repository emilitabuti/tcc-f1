# Roadmap — Estado Atual e Próximas Etapas

Data: 02/06/2026

---

## Resumo executivo

As Semanas 1 e 2 do cronograma estão **completamente executadas e documentadas**. O pipeline de dados é sólido, os leakages foram corrigidos antes do primeiro treino, e os modelos finalistas foram escolhidos com critério empírico rastreável.

**Estado quantitativo:**

| Métrica | Meta | Resultado | Status |
|---|---|---|---|
| MAE (LightGBM) | ≤ 2.5 | 2.313 | ✅ |
| MAE (Random Forest) | ≤ 2.5 | 2.328 | ✅ |
| RMSE | ≤ 3.0 | 3.008 / 3.020 | ⚠️ Marginalmente acima |
| R² | ≥ 0.75 | 0.660 / 0.657 | ❌ Abaixo da meta |
| Kendall τ | ≥ 0.60 | 0.655 / 0.651 | ✅ |
| Top-3 accuracy | ≥ 70% | 24% / 21% | ❌ Métrica de regressão vs. classificação |

**Algoritmos finalistas:** LightGBM + Random Forest  
**Baseline:** Ridge Regression (melhor MAE global: 2.273)  
**Dataset:** 2.943 linhas, 15 features, 0 NaN, 0 colunas proibidas

---

## Riscos ativos

Ordenados por impacto potencial na defesa ou nos próximos resultados.

### 1. R² abaixo de 0.75

**Impacto:** meta declarada na arquitetura não foi atingida.  
**Argumento preparado:** a meta vem de TabNet [7] com dados e features diferentes. O melhor fold individual (Ridge, 2024) atinge 0.714. A degradação em 2025 é o fenômeno de drift que a Fase 2 endereça. Documentado em `docs/tecnico/10_resultados_feature_importance.md`.  
**Ação:** nenhuma correção necessária — argumento de defesa está pronto.

### 2. Mapeamento `race_name → circuit_id` hardcoded

**Impacto:** Madrid 2026 (presente nos raw data) não está no mapa — o pipeline quebraria ao processar esse GP.  
**Arquivo:** `src/09_preparar_base_feature_engineering.py` — dicionário interno de mapeamento.  
**Ação antes da Semana 3:** adicionar Madrid ao mapeamento antes de rodar o pipeline com dados 2026.

### 3. `coef_pilotos.csv` — cópia legada referenciada por código ativo

**Impacto:** `src/feature_engineering_parte_1.py` lê `coef_pilotos.csv` (linhas 24-25). Se o arquivo for removido, o pipeline quebra.  
**Ação:** atualizar as linhas 24-25 para ler `coef_pilotos_rapm_2018_2025.csv` e depois remover a cópia legada. Baixo risco, mas deve ser feito antes de qualquer limpeza adicional.

### 4. Alpha=10.0 do RAPM sem validação empírica

**Impacto:** coeficientes podem ser subótimos para entidades com poucas corridas.  
**Ação:** documentado como limitação em `docs/tecnico/05_rapm_ridge.md`. Para a defesa: alpha conservador é defensável em séries curtas. Não requer correção antes da Semana 3.

### 5. Pilotos novos de 2026 sem mapeamento de driver_number

**Impacto:** `update_openf1_2026.py` usa `driver_3_2026`, `driver_11_2026`, etc. como IDs para os 4 pilotos novos. Features RAPM e DNF rates para esses pilotos ficam em cold-start (0.0).  
**Ação antes da análise de drift:** identificar os 4 pilotos pelos números 3, 11, 41, 77 e atualizar `DRIVER_NUMBER_TO_ID_2026_NEW` em `src/update_openf1_2026.py`.

---

## O que está pronto para a Semana 3

### Infraestrutura

- [x] Dataset modelagem 2018-2025 com 15 features, 0 NaN
- [x] LightGBM e RF treinados com hiperparâmetros ótimos (Optuna)
- [x] 4 corridas de 2026 processadas em `openf1_2026_available.csv` (76 linhas, 15 features)
- [x] `feature_importance_2024.csv` — referência pré-regulamento para comparação de drift
- [x] `update_openf1_2026.py` funcional para extração incremental

### Documentação

- [x] 10 documentos técnicos em `docs/tecnico/`
- [x] Validação metodológica com evidências em `docs/validacao_metodologica.md`
- [x] Inventário completo em `docs/inventario/`
- [x] Registro de limpeza em `docs/registro_limpeza.md`

---

## Plano para a Semana 3 (31/05 a 05/06)

Organizado por responsabilidade, conforme o cronograma `Cronograma_Revisado_LightGBM.pdf`.

### P1 — Documentação do Código

| Dia | Tarefa | Dependência |
|---|---|---|
| Segunda 01/06 *(já passado)* | Criar `update_openf1_2026.py` | ✅ Feito |
| Terça 01/06 | Documentar `pipeline_dados.py`, `rapm_ridge.py` com docstrings | Nenhuma |
| Terça 01/06 | Documentar `tuning_lightgbm.py` — diferenças em relação ao XGBoost | Nenhuma |
| Quarta 02/06 | Criar notebook de demonstração dos 2 finalistas | Modelos tunados já disponíveis |
| Quinta 03/06 | Criar README com instruções de reprodução + seção dos 3 algoritmos | Notebook pronto |
| Sexta 04/06 | Revisão cruzada: código vs. texto do P3 | P3 disponível |

### P2 — Análises Finais e Visualizações

| Dia | Tarefa | Dependência |
|---|---|---|
| Segunda 01/06 *(já passado)* | Extrair 2026 via OpenF1 | ✅ Feito (4 corridas disponíveis) |
| **Antes de prosseguir** | Adicionar Madrid 2026 ao mapa de circuitos | `09_preparar_base_feature_engineering.py` |
| **Antes de prosseguir** | Identificar pilotos novos (nºs 3, 11, 41, 77) | `update_openf1_2026.py` |
| Terça 01/06 | Aplicar LightGBM e RF nas 4 corridas de 2026 — curva de drift | Dados 2026 prontos |
| Terça 01/06 | Análise por tipo de circuito (urbano vs. permanente) | Dados de modelagem 2025 |
| Quarta 02/06 | Feature importance 2024 vs. primeiras corridas 2026 | feature_importance_2024.csv |
| Quinta 03/06 | Comparação degradação: LightGBM vs. RF em 2026 | Predições 2026 |
| Sexta 04/06 | Finalizar visualizações em qualidade de publicação | Todas as análises prontas |

### P3 — Resultados Parciais + Próximos Passos (TCC)

| Dia | Tarefa | Dependência |
|---|---|---|
| Segunda 01/06 *(já passado)* | Seção Resultados — tabela comparativa 4 modelos | ✅ Dados prontos |
| Terça 01/06 | Seção Resultados — feature importance | ✅ Dados prontos |
| Quarta 02/06 | Seção Resultados — drift 2026 + análise contrafactual | P2 análise de drift |
| Quinta 03/06 | Próximos Passos — arquitetura Fase 2 com TrAdaBoost | Cronograma Fase 2 |
| Sexta 04/06 | Integrar todas as seções em documento ABNT único | Todas as seções prontas |

---

## Checklist antes de qualquer execução da Semana 3

- [ ] Identificar pilotos com números 3, 11, 41, 77 no grid 2026
- [ ] Atualizar `DRIVER_NUMBER_TO_ID_2026_NEW` em `src/update_openf1_2026.py`
- [ ] Adicionar Madrid 2026 (`circuit_id`: `madring`) no mapa de circuitos de `09_preparar_base_feature_engineering.py`
- [ ] Re-rodar `update_openf1_2026.py` após as correções acima
- [ ] Confirmar que `openf1_2026_available.csv` tem `driver_id` reais (não `driver_X_2026`)

---

## Lacunas de documentação que ainda precisam ser preenchidas no TCC

Estes itens não foram criados nesta auditoria mas precisam existir no documento final:

| Item | Onde incluir no TCC | Urgência |
|---|---|---|
| Argumento formal: por que Ridge supera árvores em MAE | Seção Resultados / Discussão | Alta |
| Tabela de comparação de MAE com a literatura (contextualizando 2018+ vs. 2014+) | Seção Resultados | Alta |
| Documentação RAPM como Fixed Effects (não RAPM estrito) | Seção Metodologia — RAPM | Média |
| Justificativa para decay=0.95 vs. 0.75 do paper | Seção Metodologia — Walk-Forward | Média |
| Análise de viés de sobrevivência dos DNFs | Seção Limitações | Média |
| Por que top-3 accuracy 18-24% é esperado (regressão vs. classificação) | Seção Resultados | Alta |

---

## Fase 2 — o que está preparado

A infraestrutura para a Fase 2 (TrAdaBoost + análise de drift) está parcialmente pronta:

| Artefato | Status |
|---|---|
| 4 corridas de 2026 com 15 features | ✅ `openf1_2026_available.csv` |
| Referência de feature importance pré-2026 | ✅ `feature_importance_2024.csv` |
| Biblioteca `adapt==0.4.5` (TrAdaBoost) | ✅ No `requirements.txt` |
| Modelos finalistas com hiperparâmetros ótimos | ✅ Salvos em `optuna_*_best_params.json` |
| `update_openf1_2026.py` para extração incremental | ✅ Funcional (4/24 corridas) |

O que falta para a Fase 2: dados de corridas 4 a 24 de 2026 conforme o calendário avança ao longo da temporada.
