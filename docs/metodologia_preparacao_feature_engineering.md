# Preparacao da base para Feature Engineering

## Recorte oficial

O recorte temporal oficial do projeto e **2018 em diante**.

Justificativa: o corte em 2018 coincide com a introducao do sistema de Power Unit hibrido
de forma consolidada e com a disponibilidade consistente de dados via FastF1 (laps, setores,
compostos de pneu, TrackStatus). O paper de Thomas et al. (2021) — referencia [7] do TCC —
justifica recortes temporais pela homogeneidade regulatoria, criterio que orienta a escolha
de 2018 como inicio efetivo. Mencoes a 2014 em versoes anteriores da documentacao estavam
desatualizadas e foram corrigidas.

## Base oficial

A base oficial para a etapa de Feature Engineering e:

- `data/processed/dataset_feature_engineering_ready_2018_2025.csv`

Para experimentos que precisam treinar apenas ate 2024, usar:

- `data/processed/dataset_feature_engineering_ready_2018_2024.csv`

## Anti-leakage

O target do problema e `finish_position`.

As seguintes colunas nunca devem entrar em `X`:

- `finish_position`
- `points`
- `race_points`
- `fastest_lap_race`
- `previous_position`

`finish_position` permanece na base pronta para Feature Engineering porque e necessario como
target e como historico para features causais (recent_form, vitorias acumuladas, sinergia
piloto-construtor). Antes da modelagem, ele deve ser separado de `X`.

`points` foi removido da base pronta por ser uma variavel pos-corrida altamente derivada do
resultado — inclui-la em `X` configuraria data leakage direto.

## Qualifying: decisao metodologica

A arquitetura original previa imputacao KNN para valores ausentes de qualifying. Esta etapa
**nao foi aplicada** pela seguinte razao: a feature final que entra no modelo e
`qualifying_position` (posicao numerica de largada apos qualifying), nao os tempos Q1/Q2/Q3.
Para os 18 registros (~0,6% da base) sem posicao de qualifying disponivel, foi usado
`grid_position` como proxy conservador, com `grid_penalty=0` quando a penalidade nao era
conhecida. Q1, Q2 e Q3 permanecem nos dados brutos para auditoria.

Esta decisao e metodologicamente equivalente ao tratamento aplicado por Koopman (ref. [5])
para corridas onde a posicao de qualifying nao estava disponivel.

## Track complexity com incidentes historicos

A feature `track_complexity` foi enriquecida nesta etapa com um componente causal de taxa
historica de Safety Car e Virtual Safety Car por circuito.

Formula final (5 componentes):

  track_complexity = 0.35 * corners_norm
                   + 0.25 * length_km_norm
                   + 0.20 * altitude_norm
                   + 0.10 * circuit_type
                   + 0.10 * incident_rate_hist_norm

onde `incident_rate_hist_norm` e a taxa historica de SC/VSC no circuito, calculada
causalmente: para cada corrida r, so usa corridas anteriores a r no mesmo circuito
(expanding().mean().shift(1)). Cold-start usa a taxa global de 2018-2024.

A versao estatica original (sem incidentes) fica preservada em `track_complexity_static`
para auditoria e comparacao de importancia de feature.

Esta implementacao esta alinhada com a especificacao da arquitetura (ref. Ruan et al. [2]
e Barra et al. [3]) que citam incidentes historicos como componente de complexidade de pista.

## Pit stops sem vazamento temporal

A coluna `avg_pit_stops_circuit` foi recalculada na base pronta usando apenas corridas
anteriores do mesmo circuito. A media global por circuito produzida na etapa 07 foi
preservada como `avg_pit_stops_circuit_static_global` apenas para auditoria.

## Outliers — estado final

Foram aplicadas duas reconciliacoes em sequencia:

1. **Pos-contexto SC/VSC (etapa 09, passa 1):** 13 registros com `outlier_revisao` e
   `safety_car_flag=1` foram promovidos a `outlier_legitimo`. A propria regra metodologica
   estabelece que extremos em corridas com Safety Car sao eventos reais de pista.

2. **Colunas nao-feature (etapa 09, pass 2):** outliers cujas colunas anomalas pertencem
   exclusivamente a `OUTLIER_COLS_NAO_FEATURE` (tempos FastF1 que nao entram em X) foram
   promovidos a `outlier_legitimo`. A linha de resultado da corrida e valida; apenas o
   tempo FastF1 apresentou valor extremo.

Caso especifico resolvido: Stroll, GP da Estira 2021 (round 8), `fastf1_avg_sector1`
elevado, sem safety car. Conclusao: setor 1 elevado reflete provavelmente dano mecanico
leve ou percurso sob bandeira amarela local — o resultado (8o lugar, +1 Lap) e valido.
A coluna `fastf1_avg_sector1` nao entra em X. Reclassificado como `outlier_legitimo`.

Estado final: 0 `outlier_revisao`. Todos os outliers detectados tem classificacao
definitiva.

## DNF rates

As features `driver_dnf_rate` e `constructor_dnf_rate` nao devem ser calculadas a partir
da base FE-ready, pois ela segue DNF Excluded.

A fonte obrigatoria para essas taxas e:

- `data/processed/historico_dnf_classificado_2018_2025.csv`

As taxas devem ser causais: usar apenas corridas anteriores a corrida alvo com shift(1)
dentro de cada grupo (driver_id ou constructor_id).

## Normalizacao na modelagem

Em walk-forward, scalers devem ser ajustados apenas no treino de cada fold e aplicados na
validacao correspondente. As colunas normalizadas existentes no CSV sao artefatos de
preprocessamento/auditoria e nao substituem o fit temporal dentro da modelagem.

## Arquivos gerados

- `dataset_feature_engineering_ready_2018_2024.csv`
- `dataset_feature_engineering_ready_2018_2025.csv`
- `target_finish_position_2018_2024.csv`
- `target_finish_position_2018_2025.csv`
- `outliers_revisao_2018_2025.csv`
- `manifest_feature_engineering.json`
- `relatorio_09_preparacao_feature_engineering.txt`
