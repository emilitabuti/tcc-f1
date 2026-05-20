# Analise critica pre-Feature Engineering - validacao ate quarta-feira 19/05

Data da analise: 20/05/2026  
Escopo: validacao do repositorio `/home/emili-tabuti/tcc-f1` contra `docs/ArquiteturaProposta.pdf` e `docs/Cronograma_Revisado.pdf`, considerando que o cronograma foi executado ate a etapa de quarta-feira 19/05.

> Observacao: o cronograma do PDF trata quinta-feira 20/05 como a proxima etapa. A validacao abaixo segue essa convencao interna do cronograma, independentemente do calendario real do ambiente.

> Atualizacao metodologica: o recorte oficial do projeto e **2018 em diante**. Os PDFs usados como referencia ainda citam 2014 em alguns trechos, mas essa informacao esta desatualizada. Portanto, o uso de 2018-2025 no repositorio nao deve ser tratado como lacuna tecnica; deve ser tratado como decisao metodologica atual que precisa ser refletida nos documentos formais.

> Atualizacao de implementacao: as lacunas operacionais identificadas como preparatorias para quinta-feira foram implementadas em `src/07_integrar_fontes_suporte.py`, `src/08_processar_openf1_2025.py`, `src/09_preparar_base_feature_engineering.py` e `src/pipeline_dados.py`. A base oficial para Feature Engineering passou a ser `data/processed/dataset_feature_engineering_ready_2018_2025.csv`.

## 1. Veredito executivo

O projeto tem uma base de dados funcional e auditavel para o fim da etapa de quarta-feira: existe pipeline em `src/pipeline_dados.py`, bases processadas em `data/processed/`, relatorios por etapa, ausencia de nulos no dataset final, ausencia de duplicatas por `RaceID`, tratamento de DNF aplicado, outliers marcados, e enriquecimento com clima, circuito, pit stops, qualifying e safety car parcial.

Depois das correcoes, a base esta pronta para iniciar a etapa de quinta-feira com foco em RAPM e Feature Engineering. Ainda ha tarefas da propria quinta-feira a implementar, mas os bloqueios de preparacao da base foram enderecados: anti-leakage documentado, safety car historico integrado, validacao 2025 renomeada, outliers em revisao materializados e pit stops recalculados de forma causal.

Pontos que permanecem como atencao para a quinta-feira:

1. `finish_position` permanece na base FE-ready como target/historico; deve ser removido de `X` antes da modelagem.
2. Os 14 outliers em revisao foram exportados para decisao metodologica.
3. Ainda nao existem os artefatos de quinta-feira (`rapm_ridge.py`, `coef_pilotos.csv`, `coef_construtores.csv`, `feature_engineering.py`), o que e esperado antes de iniciar a quinta.

Conclusao explicita: **a base esta pronta para avancar para quinta-feira (20/05)**. A proxima etapa deve começar por `rapm_ridge.py` e pelas features historicas causais, usando `manifest_feature_engineering.json` como contrato anti-leakage.

## 2. Evidencias tecnicas verificadas

### 2.1 Estrutura do repositorio

Arquivos centrais encontrados:

- Scripts de coleta e preprocessamento em `src/`.
- Bases brutas em `data/raw/`.
- Bases processadas e relatorios em `data/processed/`.
- Scalers em `models/preprocessing/`.
- Documentacao metodologica em `docs/`.

Pipeline principal:

- `src/pipeline_dados.py`
- Etapas declaradas: limpeza, DNF, encoding, normalizacao, ausentes, outliers, integracao de fontes, dataset 2025.

Validacao executada:

- `python3 -m py_compile src/*.py`: sem erro de sintaxe.
- `python3 src/pipeline_dados.py --only 7 8 9`: etapas 07, 08 e 09 executaram com sucesso.

### 2.2 Dimensoes e cobertura das bases

Base final principal:

- `data/processed/dataset_feature_engineering_ready_2018_2025.csv`
- Dimensao: 2943 linhas x 117 colunas.
- Temporadas: 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025.
- Corridas por temporada:
  - 2018: 21
  - 2019: 21
  - 2020: 17
  - 2021: 22
  - 2022: 22
  - 2023: 22
  - 2024: 24
  - 2025: 24
- `RaceID`: 0 nulos, 0 duplicatas.
- Colunas criticas (`season`, `round`, `driver_id`, `constructor_id`, `grid_position`, `finish_position`, `laps`, `RaceID`, `race_name`, `circuit_id`): 0 nulos.
- Total de nulos no dataset final: 0.
- `points`: removido da base FE-ready.
- `finish_position`: preservado como target/historico, proibido em `X` pelo manifest.
- `avg_pit_stops_circuit`: recalculado de forma causal, usando apenas corridas anteriores.

Base 2025 isolada:

- `data/processed/validacao_2025_clean.csv`
- Dimensao: 419 linhas x 114 colunas.
- Corridas: 24.
- Pilotos unicos: 21.
- Construtores unicos: 10.
- `RaceID`: 0 nulos, 0 duplicatas.
- Total de nulos: 0.
- Alias legado mantido: `data/processed/openf1_2025_clean.csv`.

### 2.3 DNF

Tratamento aplicado:

- Variante adotada: DNF Excluded.
- Base 2018-2025 antes do DNF Excluded: 3458 registros.
- Base 2018-2025 apos DNF Excluded: 2943 registros.
- Removidos: 515 registros.

Remocao por temporada:

| Temporada | Registros brutos | Registros finais | Removidos | Percentual removido |
|---|---:|---:|---:|---:|
| 2018 | 420 | 335 | 85 | 20,2% |
| 2019 | 420 | 360 | 60 | 14,3% |
| 2020 | 340 | 283 | 57 | 16,8% |
| 2021 | 440 | 381 | 59 | 13,4% |
| 2022 | 440 | 366 | 74 | 16,8% |
| 2023 | 440 | 374 | 66 | 15,0% |
| 2024 | 479 | 425 | 54 | 11,3% |
| 2025 | 479 | 419 | 60 | 12,5% |

Diagnostico: a decisao esta alinhada ao uso do benchmark RAPM citado pela arquitetura, mas e necessario preservar a base classificada para calcular `driver_dnf_rate` e `constructor_dnf_rate` na Feature Engineering. A base DNF Excluded, sozinha, zera as flags de DNF e nao permite calcular taxas historicas corretamente.

### 2.4 Ausentes

Relatorio `relatorio_05_tratamento_valores_ausentes.txt`:

- Tempos e setores: 0 nulos antes e depois nas colunas avaliadas.
- Composto: 0 nulos antes e depois.
- Qualifying KNN: nao aplicado nessa etapa porque colunas de qualifying ainda nao existiam nesse ponto.

Diagnostico: a base final esta sem nulos, mas a imputacao de qualifying por KNN nao foi exercitada no fluxo principal de quarta. O qualifying foi integrado posteriormente na etapa 07 via `fastf1_qualifying_2018_2025.csv`, com proxy `grid_position` quando faltante.

### 2.5 Outliers

Relatorio `relatorio_06_tratamento_outliers.txt`:

- 2018-2024:
  - `nao_outlier`: 2510
  - `outlier_revisao`: 10
  - `outlier_legitimo`: 4
  - removidos: 0
- 2018-2025:
  - `nao_outlier`: 2917
  - `outlier_revisao`: 14
  - `outlier_legitimo`: 12
  - removidos: 0

Diagnostico: a politica de nao remover automaticamente extremos plausiveis e defensavel para F1. Porem, outliers em revisao precisam de decisao antes da modelagem, porque podem distorcer features de ritmo, pit stop e possivelmente RAPM se usados sem tratamento.

### 2.6 Integracao de fontes de suporte

Relatorio `relatorio_07_integracao_fontes.txt`:

- `grid_position` 0 corrigido para 21:
  - 2018-2024: 35 registros.
  - 2018-2025: 35 registros.
- `qualifying_position`: cobertura final 100%, mas com proxy `grid_position` nos ausentes.
- Circuitos integrados: 32.
- `weather_impact_factor`: calculado e sem nulos.
- `avg_pit_stops_circuit`: calculado e sem nulos.
- `safety_car_flag`:
  - 2018-2024: todos 0 por limitacao de fonte.
  - 2025: 202 registros com safety car.

Diagnostico atualizado: a integracao de safety car deixou de ser assimetrica. A fonte principal agora e FastF1 `TrackStatus` para 2018-2025, com OpenF1 Race Control como corroboracao adicional em 2025.

## 3. Avaliacao do que deveria estar concluido ate quarta-feira 19/05

### 3.1 Segunda-feira 17/05

| Entrega esperada | Estado atual | Avaliacao |
|---|---|---|
| Ambiente com dependencias fixadas | `requirements.txt` existe com pandas, numpy, scikit-learn, xgboost, lightgbm, fastf1, optuna, adapt, scipy, matplotlib, seaborn | Concluido |
| Ergast/Jolpica 2018-2024 | Implementado 2018-2024 | Concluido |
| FastF1 laps/qualifying/compostos | Laps e qualifying 2018-2025 presentes | Concluido no recorte 2018+ |
| OpenF1 validada | Arquivos OpenF1 existem e ha documento de mapeamento | Parcial |
| Mapeamento OpenF1 -> Ergast | `docs/mapeamento_openf1_ergast.md` existe | Concluido parcialmente |
| Completude documentada | `data/raw/dados_ausentes.txt` existe, relatorios por etapa existem | Parcial |

### 3.2 Terca-feira 18/05

| Entrega esperada | Estado atual | Avaliacao |
|---|---|---|
| Limpeza inicial | `limpeza_ergast_fastf1.py` gera bases limpas e relatorio | Concluido |
| RaceID | Criado como `driver_id + season + round`, 0 duplicatas | Concluido |
| Filtro temporal oficial 2018+ | Implementado como 2018+ | Concluido |
| DNF Excluded | Implementado e documentado | Concluido |
| Encoding | One-hot circuito/construtor e ordinal composto | Concluido |
| Normalizacao | Z-score e MinMax com scalers salvos | Concluido |

### 3.3 Quarta-feira 19/05

| Entrega esperada | Estado atual | Avaliacao |
|---|---|---|
| Tratamento de valores ausentes | Implementado; base final sem nulos | Concluido |
| Tempos por mediana circuito/ano | Implementado | Concluido |
| Composto por moda da corrida | Implementado | Concluido |
| Qualifying KNN | Previsto, mas nao exercitado; qualifying integrado depois com proxy | Parcial |
| Outliers > 3 desvios por circuito | Implementado | Concluido com ressalvas |
| Manter outliers legitimos com flag | Implementado; safety car historico ausente | Parcial |
| Remover espurios | Nenhum espurio removido | Aceitavel, mas exige revisao |
| Dataset limpo entregue | `dataset_feature_engineering_ready_2018_2025.csv` existe | Concluido |

## 4. Relacao com a arquitetura proposta

### 4.1 Aderencias

- A chave `RaceID` foi criada e nao apresenta duplicatas.
- `grid_position` e `finish_position` foram validados sem nulos.
- DNF Excluded foi implementado e documentado.
- One-hot para circuito e construtor foi aplicado.
- Encoding ordinal de composto foi aplicado.
- Normalizacao por Z-score e MinMax foi aplicada.
- Outliers usam criterio de 3 desvios padrao por circuito.
- Dataset final contem variaveis de suporte esperadas para FE:
  - `circuit_type`
  - `track_complexity`
  - `altitude_m`
  - `weather_impact_factor`
  - `avg_pit_stops_circuit`
  - `safety_car_flag`
  - `qualifying_position`
  - `grid_penalty`

### 4.2 Divergencias e alinhamentos documentais

| Arquitetura | Implementacao atual | Risco |
|---|---|---|
| Corte 2018 em diante | Corte 2018 em diante | Aderente ao recorte oficial atualizado; PDFs precisam ser revisados para remover mencoes antigas a 2014 |
| OpenF1 para validacao/alimentacao fase 2 | `validacao_2025_clean.csv` e recorte documentado do dataset processado; alias legado `openf1_2025_clean.csv` mantido | Risco controlado por documentacao |
| Safety car flag historica | 2018-2025 via FastF1 TrackStatus; 2025 corroborado por OpenF1 | Resolvido |
| Qualifying 0,07% ausente com KNN | KNN nao aplicado; etapa posterior usa proxy grid_position | Divergencia metodologica |
| Features finais sem leakage | `points` removido da base FE-ready; manifest proibe `finish_position` em X | Resolvido para a preparacao da base |
| `tire_compound_start` | Criada na etapa 09 a partir de `compound_ordinal` | Resolvido |
| `season_factor` | Criada na etapa 09 a partir de `season` | Resolvido |
| RAPM | Ainda nao implementado | Bloqueia features centrais de quinta |

## 5. Relacao com as referencias bibliograficas da arquitetura

### 5.1 RAPM, walk-forward e time-decay - refs. [9], [10], [18]

O projeto ainda nao tem `rapm_ridge.py`, matriz esparsa piloto/construtor, Ridge iterativo ou pesos temporais. Isso e esperado antes de quinta, mas a base precisa estar pronta para esse script.

Diagnostico: com o recorte oficial 2018-2025, o time-decay e o RAPM terao oito temporadas disponiveis na base atual. Isso e suficiente para iniciar a modelagem auxiliar, desde que a metodologia explique que 2018 foi escolhido como inicio efetivo por consistencia, disponibilidade e qualidade das fontes integradas. Os PDFs devem ser atualizados para nao sugerirem 2014 como requisito vigente.

### 5.2 Feature Engineering em F1 - refs. [2], [3], [6], [8]

As features de circuito, clima e pit stops ja comecaram a aparecer. Isso e positivo. Porem:

- `track_complexity` ainda usa comprimento, curvas, altitude e tipo de circuito, mas nao incidentes historicos, embora a arquitetura cite incidentes.
- `weather_impact_factor` usa temperatura, umidade e chuva; nao inclui track temperature ou vento, embora `dados_necessarios.txt` cite esses campos como complementares.
- `avg_pit_stops_circuit` foi recalculada na etapa 09 de forma causal, usando apenas corridas anteriores do mesmo circuito. A media global antiga foi preservada como `avg_pit_stops_circuit_static_global` apenas para auditoria.

### 5.3 Predicao genuina e data leakage - refs. [2], [3], [4], [5]

A arquitetura e os estudos do Grupo 2 exigem evitar variaveis pos-corrida. Na base FE-ready, `points` foi removido e `finish_position` permanece apenas como target/historico. O manifest explicita que `finish_position` nunca deve entrar em `X`.

Risco controlado: `points` nao esta mais na base FE-ready; `finish_position` ainda exige separacao explicita antes da modelagem.

### 5.4 Concept drift e transferencia - refs. [13], [14], [15], [21]

A base precisa preservar comparabilidade entre 2025 e 2026. A integracao de `safety_car_flag` por FastF1 `TrackStatus` reduziu o risco de drift artificial por fonte, porque agora a cobertura historica 2018-2025 e consistente.

## 6. Lacunas encontradas, impacto e prioridade

| ID | Lacuna | Evidencia | Impacto | Prioridade | Acao recomendada |
|---|---|---|---|---|---|
| L1 | PDFs/documentos ainda citam 2014 como corte | ArquiteturaProposta.pdf menciona 2014; implementacao oficial usa 2018 | Inconsistencia documental, nao tecnica | P1 | Atualizar arquitetura/metodologia para declarar 2018 como recorte oficial |
| L2 | Colunas pos-corrida presentes na base pre-features | `points` removido da base FE-ready; manifest proibe `finish_position` em X | Risco controlado | Resolvido | Usar `manifest_feature_engineering.json` antes de modelar |
| L3 | `openf1_2025_clean.csv` nao e OpenF1-first | `validacao_2025_clean.csv` criado como nome principal; alias legado mantido | Risco controlado | Resolvido | Atualizar referencias futuras para `validacao_2025_clean.csv` |
| L4 | Safety car historico ausente | FastF1 TrackStatus integrado para 2018-2025 | Resolvido | Resolvido | Manter regra documentada no relatorio 07 |
| L5 | Outliers em revisao sem decisao final | `outliers_revisao_2018_2025.csv` gerado com 14 casos | Decisao metodologica pendente, mas rastreada | P1 | Decidir manter/remover/winsorizar antes da modelagem |
| L6 | `avg_pit_stops_circuit` possivelmente usa futuro | Recalculada na etapa 09 com apenas corridas anteriores; media global antiga preservada para auditoria | Resolvido | Resolvido | Usar a coluna FE-ready, nao o arquivo da etapa 07 direto |
| L7 | `weather_impact_factor` pode usar parametros globais e formula simplificada | Normalizacao baseada em treino 2018-2024; formula sem track temp/vento | Aceitavel, mas precisa justificativa e evitar uso de futuro em folds | P2 | Congelar formula, documentar, e recalcular por fold se necessario |
| L8 | `track_complexity` incompleta frente a arquitetura | Nao ha incidentes historicos no score | Feature menos aderente a RF+SHAP/arquitetura | P2 | Adicionar incidente historico acumulado ou renomear como `track_static_complexity` |
| L9 | Qualifying KNN nao exercitado | Relatorio 05: `aplicado=False`; etapa 07 usa proxy | Divergencia da metodologia | P2 | Documentar que KNN nao foi necessario/nao aplicavel, ou integrar Q1/Q2/Q3 e testar KNN |
| L10 | Features centrais de quinta inexistentes | Nao ha `rapm_ridge.py`, `feature_engineering.py`, `coef_*.csv` | Bloqueia FE real | P0 para quinta | Implementar como primeira tarefa da quinta antes de qualquer modelo |
| L11 | DNF rates nao podem ser calculados pela base DNF Excluded | Flags DNF sao 0 no dataset final | `driver_dnf_rate` e `constructor_dnf_rate` ficariam incorretas | P0 para FE | Calcular taxas usando base classificada completa, sempre com historico anterior a corrida |
| L12 | One-hot de construtor/circuito fitado em 2018-2024 | `handle_unknown="ignore"` para 2025+ | Novas equipes/circuitos viram vetor zerado | P2 | Manter, mas documentar; para 2026 considerar categorias novas ou target/impact encoding temporal |
| L13 | Falta separacao formal entre base de treino, validacao e dataset de feature engineering | Muitos arquivos processados coexistem | Risco operacional de usar arquivo errado | P1 | Definir manifest: entrada oficial da FE, target, colunas proibidas, bases auxiliares |

## 7. Diagnostico geral da base de dados

### Pontos fortes

- Boa rastreabilidade: cada etapa gera relatorio.
- Base final sem nulos.
- Chave primaria consistente.
- DNF Excluded aplicado e documentado.
- Scalers salvos.
- One-hot e ordinal encoding implementados.
- Circuitos, clima, pit stops e qualifying ja integrados.
- Pipeline executavel nas etapas 07, 08 e 09.

### Pontos fracos

- Inconsistencia documental entre PDFs desatualizados e recorte oficial 2018+.
- PDFs/metodologia ainda precisam refletir oficialmente o recorte 2018+.
- Os 14 outliers em revisao ainda exigem decisao antes da modelagem.
- Nao existem ainda os scripts de quinta-feira para RAPM e Feature Engineering.

## 8. Checklist obrigatorio antes da Feature Engineering

### Bloqueadores P0

- [x] Criar arquivo/funcao com `TARGET = finish_position` e `COLUNAS_PROIBIDAS = [points, finish_position, race_points, fastest_lap_race, previous_position]`.
- [x] Garantir que `points` nunca entra na base FE-ready.
- [x] Corrigir nomenclatura/fonte de `openf1_2025_clean.csv`: `validacao_2025_clean.csv` agora e a saida principal.
- [ ] Implementar `rapm_ridge.py` com Ridge iterativo, matriz piloto/construtor e time-decay.
- [ ] Calcular `driver_dnf_rate` e `constructor_dnf_rate` a partir da base classificada, nao da base DNF Excluded.
- [x] Definir entrada oficial da FE: `data/processed/dataset_feature_engineering_ready_2018_2025.csv` + bases auxiliares classificadas.

### Correcoes P1

- [ ] Atualizar PDFs/metodologia para declarar 2018+ como recorte oficial.
- [x] Integrar safety car historico 2018-2024 via FastF1 `TrackStatus`.
- [ ] Revisar os 14 `outlier_revisao` e documentar decisao.
- [x] Recalcular `avg_pit_stops_circuit` de forma temporal/causal.
- [ ] Recalcular demais features historicas agregadas de forma temporal: sinergia, forma recente, vitorias, experiencia e DNF rates sempre ate a corrida anterior.
- [ ] Gerar manifest de colunas finais esperadas e colunas auxiliares.
- [x] Criar manifest de Feature Engineering com target e colunas proibidas.
- [ ] Criar teste automatico que falha se houver nulos, duplicatas de `RaceID`, colunas proibidas em `X` ou uso de futuro.

### Melhorias P2

- [x] Adicionar `season_factor`.
- [x] Padronizar `tire_compound_start` a partir de `compound_ordinal`.
- [ ] Reavaliar formula de `weather_impact_factor` incluindo `TrackTemp` e `WindSpeed` se houver justificativa.
- [ ] Enriquecer `track_complexity` com incidente historico ou renomear a feature para refletir a formula real.
- [ ] Criar README de reproducibilidade.

## 9. Recomendacoes tecnicas para quinta-feira 20/05

Ordem recomendada:

1. Usar a base oficial `dataset_feature_engineering_ready_2018_2025.csv` e o manifest de colunas.
2. Registrar no texto metodologico que 2018+ e o recorte oficial.
3. Respeitar `manifest_feature_engineering.json` antes de qualquer feature/modelagem.
4. Implementar RAPM Ridge em script separado.
5. Implementar FE temporal com `groupby` ordenado por `season`, `round`, sempre usando `shift(1)` ou treino ate corrida anterior.
6. Calcular DNF rates na base classificada completa.
7. Usar `avg_pit_stops_circuit` da base FE-ready, ja recalculado de forma causal.
8. Rodar validacao automatica de nulos, duplicatas, leakage e cobertura por temporada.

## 10. Avaliacao explicita de prontidao

### O que esta pronto

A base esta suficientemente limpa para servir como materia-prima inicial da Feature Engineering:

- sem nulos;
- sem duplicatas por `RaceID`;
- com target presente;
- com dados historicos de resultado, grid, construtor, piloto, volta, pneu, circuito, clima e pit stop;
- com DNF Excluded aplicado;
- com relatorios de auditoria.

### O que nao esta pronto

A base esta pronta para gerar features finais, com os seguintes cuidados:

- ha inconsistencia documental entre PDFs antigos e recorte oficial 2018+;
- `finish_position` deve ser separado de `X`;
- as features historicas de quinta precisam ser calculadas causalmente;
- ha outliers em revisao;
- ha ausencia total dos coeficientes RAPM.

### Veredito final

**Projeto valido ate quarta-feira 19/05 e base pronta para avancar para quinta-feira 20/05.**

Recomendacao: iniciar quinta-feira 20/05 por RAPM e Feature Engineering temporal, usando `dataset_feature_engineering_ready_2018_2025.csv`, `target_finish_position_2018_2025.csv`, `outliers_revisao_2018_2025.csv` e `manifest_feature_engineering.json` como artefatos oficiais.
