# Normalização das variáveis numéricas

## Objetivo

Esta etapa tem como objetivo padronizar variáveis numéricas para uso em modelos de Machine Learning.

A normalização evita que variáveis com escalas muito diferentes tenham impacto desproporcional no treinamento do modelo.

## Z-score

Foi aplicado Z-score, também conhecido como padronização, nas variáveis numéricas contínuas.

A fórmula geral é:

z = (x - média) / desvio padrão

As colunas normalizadas por Z-score foram:

- `fastf1_laps_count`
- `fastf1_avg_lap_time`
- `fastf1_best_lap_time`
- `fastf1_avg_sector1`
- `fastf1_avg_sector2`
- `fastf1_avg_sector3`
- `fastf1_max_tyre_life`
- `fastf1_stints_count`
- `fastf1_pit_in_count`
- `fastf1_pit_out_count`

Para preservar os dados originais, foram criadas novas colunas com o sufixo `_zscore`.

## MinMaxScaler

Foi aplicado MinMaxScaler nas variáveis `grid_position` e `laps`.

A fórmula geral é:

x_normalizado = (x - mínimo) / (máximo - mínimo)

As colunas normalizadas por MinMaxScaler foram:

- `grid_position`
- `laps`

Para preservar os dados originais, foram criadas novas colunas com o sufixo `_minmax`.

## Critério metodológico

Os parâmetros de normalização foram ajustados com base na base histórica de 2018 a 2024 e aplicados também à base 2018 a 2025.

Essa decisão evita vazamento de informação da base com 2025 para o processo de ajuste dos scalers.

## Arquivos gerados

- `historico_normalizado_2018_2024.csv`
- `historico_normalizado_2018_2025.csv`

A base principal recomendada para treinamento inicial do modelo é a versão 2018-2024.
