# Metodologia de Feature Engineering

## Escopo desta etapa

Esta etapa fecha as features historicas causais previstas na arquitetura para a Fase 1,
sem executar ainda as tarefas pos-feature engineering do fim de semana, como correlacao,
multicolinearidade e congelamento dos datasets `X/y`.

Arquivo principal gerado:

- `data/processed/dataset_features_final_2018_2025.csv`

Arquivo auxiliar de treino historico:

- `data/processed/dataset_feature_engineering_parte_1_2018_2024.csv`

## Base de entrada

Base modelavel:

- `data/processed/dataset_feature_engineering_ready_2018_2025.csv`

Base obrigatoria para DNF rates:

- `data/processed/historico_dnf_classificado_2018_2025.csv`

A base modelavel segue DNF Excluded. Por isso, `driver_dnf_rate` e
`constructor_dnf_rate` sao calculadas primeiro na base classificada de DNF e so depois
mescladas na base modelavel.

## Referencias da arquitetura

| Decisao | Referencias |
|---|---|
| Features especificas de forma, sinergia e risco historico | Ruan et al. [2], Barra et al. [3], Heilmeier et al. [6] |
| Coeficientes RAPM de piloto e construtor | Henderson et al. [9], Snoeks [10] |
| Ordem temporal e ausencia de leakage | Henderson et al. [9], Tan et al. [18] |
| Manter interpretabilidade das features | Ruan et al. [2], Barra et al. [3], Chen & Guestrin [19], Breiman [20] |

## Features calculadas

### `driver_coef_rapm` e `constructor_coef_rapm`

Coeficientes importados dos arquivos RAPM e integrados por `RaceID`.

Cold start:

- valores ausentes sao preenchidos com `0.0`.

### `recent_form_5` e `recent_form_3`

Media ponderada causal das ultimas posicoes finais do piloto.

- `recent_form_5`: ultimas 5 corridas, com maior peso para a mais recente.
- `recent_form_3`: ultimas 3 corridas, com maior peso para a mais recente.

A feature usa `finish_position` historico, nao a corrida alvo. Menor valor significa melhor
forma recente, conforme a definicao textual da arquitetura.

Cold start:

- sem historico anterior: `0.0`, marcado por `recent_form_cold_start_flag = 1`.

### `driver_experience`

Total causal de corridas anteriores do piloto.

Formula:

- `groupby("driver_id").cumcount()`

### `driver_wins_total`

Total causal de vitorias anteriores do piloto.

Regra:

- a vitoria da corrida alvo nao entra no proprio acumulado.

### `constructor_wins_total`

Total causal de vitorias anteriores do construtor.

Implementacao:

1. Agrega em nivel `constructor_id`, `season`, `round`.
2. Marca se o construtor venceu aquela corrida.
3. Aplica acumulado com deslocamento temporal.
4. Mescla de volta para as linhas piloto-corrida.

Essa implementacao evita vazamento entre os dois carros da mesma equipe na mesma corrida.

### `driver_dnf_rate`

Taxa historica causal de DNF atribuivel ao piloto.

Fonte:

- `data/processed/historico_dnf_classificado_2018_2025.csv`

Formula:

- DNFs de piloto anteriores / largadas anteriores do piloto.

Cold start:

- sem historico anterior: `0.0`.

### `constructor_dnf_rate`

Taxa historica causal de DNF mecanico do construtor.

Fonte:

- `data/processed/historico_dnf_classificado_2018_2025.csv`

Formula:

- DNFs mecanicos anteriores / entradas anteriores do construtor.

O calculo e feito em nivel corrida-construtor antes do merge para evitar que falhas da corrida
alvo contaminem as linhas dos pilotos da propria corrida.

### `driver_constructor_synergy`

Media historica causal do desempenho do piloto com o construtor.

Formula:

- media expandida de `-finish_position` por par `driver_id`, `constructor_id`, com
  deslocamento temporal.

Interpretacao:

- maior valor indica melhor sinergia historica.

Cold start:

- sem historico anterior do par: `0.0`.

## Features preservadas

As seguintes features ja estavam prontas na base FE-ready e foram preservadas:

- `grid_position`
- `qualifying_position`
- `grid_penalty`
- `circuit_type`
- `track_complexity`
- `altitude_m`
- `tire_compound_start`
- `avg_pit_stops_circuit`
- `season_factor`
- `weather_impact_factor`
- `safety_car_flag`

## Validacoes realizadas

Validacoes registradas em:

- `data/processed/relatorio_feature_engineering.txt`
- `data/processed/relatorio_11_feature_engineering_parte_1.txt`

Critérios:

- mesma quantidade de linhas entre entrada e saida;
- `RaceID` sem duplicatas;
- temporadas esperadas no arquivo;
- features finais sem NaN;
- `driver_dnf_rate` e `constructor_dnf_rate` com valores nao triviais;
- colunas proibidas por leakage nao adicionadas ao dataset final.

## Fechamento pos-feature engineering

A etapa de fim de semana foi executada apos o fechamento do dataset final de features.

Arquivos gerados:

- `data/processed/dataset_features_final_2018_2025_sem_nan.csv`
- `reports/correlacao_features/matriz_correlacao_features.csv`
- `reports/correlacao_features/correlation_matrix_features.png`
- `reports/correlacao_features/correlation_with_target.csv`
- `reports/correlacao_features/pares_correlacao_alta_maior_085.csv`
- `reports/correlacao_features/relatorio_correlacao_features.txt`
- `reports/correlacao_features/relatorio_correlacao.md`
- `data/processed/dataset_modelagem_X_2018_2025.csv`
- `data/processed/dataset_modelagem_y_2018_2025.csv`
- `data/processed/dataset_modelagem_X_2018_2024.csv`
- `data/processed/dataset_modelagem_y_2018_2024.csv`
- `data/processed/dataset_modelagem_2018_2025.csv`
- `data/processed/relatorio_feature_engineering_final.txt`
- `models/feature_selection/features_modelagem_2018_2025.json`
- `models/feature_selection/relatorio_13_selecao_features_modelagem.txt`

## Analise de correlacao

A matriz de correlacao foi calculada apenas sobre as features finais canonicas definidas em
`docs/lista_features_modelo.md`.

Nao foram incluidas na matriz de decisao:

- colunas de identificacao;
- target;
- one-hot intermediario de circuito/construtor;
- flags de auditoria de outlier;
- artefatos z-score/minmax;
- telemetria FastF1 usada apenas para auditoria ou baseline especifico.

Essa escolha preserva a interpretabilidade das features finais e evita que artefatos
intermediarios dominem a analise de multicolinearidade.

## Dataset de modelagem

O dataset `X` foi congelado com 21 features finais:

- `grid_position`
- `qualifying_position`
- `grid_penalty`
- `recent_form_5`
- `recent_form_3`
- `driver_coef_rapm`
- `driver_dnf_rate`
- `driver_experience`
- `driver_wins_total`
- `constructor_coef_rapm`
- `constructor_dnf_rate`
- `constructor_wins_total`
- `driver_constructor_synergy`
- `circuit_type`
- `track_complexity`
- `altitude_m`
- `tire_compound_start`
- `avg_pit_stops_circuit`
- `season_factor`
- `weather_impact_factor`
- `safety_car_flag`

O target `finish_position` fica apenas nos arquivos `y`.

Arquivos `y` preservam as chaves:

- `RaceID`
- `season`
- `round`
- `driver_id`
- `constructor_id`
- `finish_position`

## Fora do escopo desta etapa

As tarefas abaixo pertencem ao fechamento pos-feature engineering do fim de semana e serao
nao foram misturadas com a modelagem:

- treinamento de XGBoost;
- treinamento de Random Forest;
- tuning de hiperparametros;
- avaliacao walk-forward dos modelos;
- interpretabilidade SHAP dos modelos treinados.
