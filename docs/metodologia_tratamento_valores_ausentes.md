# Tratamento de valores ausentes

## Objetivo

Esta etapa tem como objetivo tratar valores ausentes antes do uso da base em modelos de Machine Learning.

## Regras adotadas

### Tempos de volta

As colunas de tempos de volta e setores foram imputadas pela mediana do circuito naquele ano.

Foram consideradas as seguintes colunas:

- fastf1_avg_lap_time
- fastf1_best_lap_time
- fastf1_avg_sector1
- fastf1_avg_sector2
- fastf1_avg_sector3

Quando não havia mediana disponível para o circuito naquele ano, foram aplicados fallbacks:

1. Mediana do ano
2. Mediana global da coluna
3. Valor 0, caso não houvesse nenhuma mediana disponível

### Composto de pneu

O composto de pneu foi imputado pela moda da corrida, considerando a combinação de `season` e `round`.

Após a imputação, a variável `compound_ordinal` foi recalculada conforme a regra:

- SOFT = 3
- MEDIUM = 2
- HARD = 1
- INTERMEDIATE/WET/UNKNOWN = 0

### Qualifying

Para variáveis de qualifying, foi prevista imputação por KNN.

Caso não existam colunas de qualifying na base, a etapa é registrada como não aplicada.

## Reprocessamento da normalização

Após a imputação, as colunas normalizadas foram recalculadas para manter consistência entre os valores originais e os valores padronizados.

Os parâmetros de normalização foram ajustados com base na base 2018-2024 e aplicados também à base 2018-2025.

## Arquivos gerados

- historico_imputado_normalizado_2018_2024.csv
- historico_imputado_normalizado_2018_2025.csv
