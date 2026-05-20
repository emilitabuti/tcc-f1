# Tratamento de outliers

## Objetivo

Esta etapa tem como objetivo identificar e tratar valores extremos nas variáveis de tempo de volta e setores.

## Critério adotado

Foi adotado o critério de valores acima de 3 desvios padrão da média por circuito.

Foram avaliadas as seguintes colunas:

- fastf1_avg_lap_time
- fastf1_best_lap_time
- fastf1_avg_sector1
- fastf1_avg_sector2
- fastf1_avg_sector3

## Classificação dos outliers

Os outliers foram classificados em tres grupos:

### Outliers legítimos

São valores extremos que podem ser explicados por eventos reais da corrida, como Safety Car, falha mecânica ou corrida com pneus WET/INTERMEDIATE.

Esses registros foram mantidos na base e marcados com flags.

### Outliers espúrios

São valores extremos tecnicamente inválidos, como tempos ausentes ou menores/iguais a zero.

Esses registros foram removidos da base tratada.

### Outliers para revisão

São valores extremos plausíveis, mas sem evidência suficiente para remoção automática.

Esses registros foram mantidos na base com flag, seguindo a decisão metodológica de não descartar eventos reais de corrida sem confirmação.

## Observação sobre Safety Car

A base atual não possui uma variável real de Safety Car integrada. Por isso, quando a coluna `safety_car_flag` não existe, ela é criada com valor 0.

Caso dados de Safety Car sejam integrados posteriormente, essa flag poderá ser usada para preservar outliers legítimos associados a esse evento.

## Reprocessamento da normalização

Após a remoção dos outliers espúrios, as colunas normalizadas foram recalculadas.

Os parâmetros de normalização foram ajustados com base na base 2018-2024 e aplicados também à base 2018-2025.

## Arquivos gerados

- historico_outliers_tratados_2018_2024.csv
- historico_outliers_tratados_2018_2025.csv
- outliers_removidos_2018_2024.csv
- outliers_removidos_2018_2025.csv
