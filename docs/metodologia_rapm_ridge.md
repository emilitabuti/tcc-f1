# Metodologia - RAPM Ridge

## Objetivo

Esta etapa cria coeficientes auxiliares de desempenho para pilotos e construtores usando uma abordagem inspirada em RAPM, com Ridge Regression e matriz esparsa binaria.

## Entrada

Arquivo padrao:

data/processed/dataset_feature_engineering_ready_2018_2025.csv

Colunas obrigatorias:

- season
- round
- RaceID
- race_name
- driver_id
- constructor_id
- finish_position

## Matriz do modelo

Para cada linha piloto-corrida, o script cria indicadores binarios para:

- piloto
- construtor

A matriz e mantida como esparsa para evitar aumento desnecessario de memoria.

## Target

O alvo usado no modelo e:

target = -finish_position

Com isso, coeficientes maiores representam associacao com melhores resultados historicos.

## Causalidade

A geracao e feita corrida a corrida.

Para cada corrida r, o modelo e treinado somente com corridas anteriores a r.

Assim, os coeficientes podem ser usados como feature historica sem vazamento de informacao futura.

Para corridas ou entidades sem historico suficiente, o coeficiente recebe cold start 0.0.

## Time-decay

O peso de cada observacao historica e calculado por distancia temporal:

peso = decay ^ distancia

Unidade padrao:

decay_unit = season

Valor de decay:

decay = 0.75

Corridas mais recentes recebem maior peso.

## Regularizacao Ridge

A regressao usa Ridge para reduzir instabilidade dos coeficientes.

Valor padrao:

alpha = 10.0

## Saidas

data/processed/coef_pilotos_rapm_2018_2025.csv
data/processed/coef_construtores_rapm_2018_2025.csv
data/processed/relatorio_10_rapm_ridge.txt
models/rapm/manifest_rapm_ridge.json

Tambem sao gravadas copias de compatibilidade em:

data/processed/coef_pilotos.csv
data/processed/coef_construtores.csv

## LOESS opcional

O script permite suavizar os coeficientes com LOESS usando:

python src/rapm_ridge.py --loess
