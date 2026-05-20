# Preparacao da base para Feature Engineering

## Recorte oficial

O recorte temporal oficial do projeto e 2018 em diante.

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

`finish_position` permanece na base pronta para Feature Engineering porque e necessario como target e como historico para features causais, como `recent_form_3`, `recent_form_5`, vitorias acumuladas e sinergia piloto-construtor. Antes da modelagem, ele deve ser separado de `X`.

`points` foi removido da base pronta por ser uma variavel pos-corrida fortemente derivada do resultado.

## Pit stops sem vazamento temporal

A coluna `avg_pit_stops_circuit` foi recalculada na base pronta usando apenas corridas anteriores do mesmo circuito. A media global por circuito produzida na etapa 07 foi preservada como `avg_pit_stops_circuit_static_global` apenas para auditoria e nao deve ser usada como feature temporal principal.

## Arquivos gerados

- `dataset_feature_engineering_ready_2018_2024.csv`
- `dataset_feature_engineering_ready_2018_2025.csv`
- `target_finish_position_2018_2024.csv`
- `target_finish_position_2018_2025.csv`
- `outliers_revisao_2018_2025.csv`
- `manifest_feature_engineering.json`
- `relatorio_09_preparacao_feature_engineering.txt`
