# Encoding das variáveis categóricas

## Objetivo

Esta etapa tem como objetivo transformar variáveis categóricas em representações numéricas, permitindo sua utilização por modelos de Machine Learning.

## One-Hot Encoding

Foi aplicado One-Hot Encoding para variáveis categóricas sem ordem natural.

As variáveis utilizadas foram:

- `race_name`: representação do circuito ou corrida.
- `constructor_id`: identificação da equipe/construtor.

O One-Hot Encoding cria uma coluna binária para cada categoria. Dessa forma, evita-se que o modelo interprete categorias nominais como se tivessem uma ordem numérica.

## Label Encoding ordinal para composto de pneu

Para o composto de pneu foi utilizado Label Encoding ordinal, pois os compostos de pista seca possuem uma relação técnica de dureza.

A coluna utilizada foi:

- `fastf1_first_compound`

A regra aplicada foi:

- SOFT = 3
- MEDIUM = 2
- HARD = 1
- INTERMEDIATE/WET/UNKNOWN = 0

A ordem adotada segue a relação:

SOFT > MEDIUM > HARD

Compostos intermediários, de chuva ou ausentes foram mantidos com valor 0, pois não seguem a mesma escala ordinal dos compostos de pista seca.

## Arquivos gerados

Esta etapa gera duas bases:

- `historico_encoded_2018_2024.csv`
- `historico_encoded_2018_2025.csv`

A base principal recomendada para treinamento inicial do modelo é a versão 2018-2024.
