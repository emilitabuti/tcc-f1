# Analise Exploratoria 2026 - Semana 3 P2

## Escopo

Esta analise executa a parte da P2 do cronograma: aplicar os modelos treinados em 2018-2025 nas corridas 2026 disponiveis e observar degradacao/erro inicial.

Importante: 2026 e usado como teste exploratorio de mudanca temporal. As metricas oficiais da Fase 1 continuam sendo os folds 2023, 2024 e 2025.

## Cobertura

- Corridas avaliadas: 7 (R1 Australian Grand Prix, R2 Chinese Grand Prix, R3 Japanese Grand Prix, R6 Miami Grand Prix, R7 Canadian Grand Prix, R8 Monaco Grand Prix, R9 Barcelona Grand Prix).
- Predicoes por modelo: 114.
- Linhas removidas por `finish_position` ausente: 9.
- Features finais: 13/13 presentes em `data/processed/openf1_2026_available.csv`.

## Resultado por modelo

| Modelo | MAE medio | RMSE medio | R2 medio | Kendall tau medio | Score exploratorio |
|---|---:|---:|---:|---:|---:|
| LightGBM | 2.4567 | 3.0059 | 0.5614 | 0.6390 | 0.5122 |
| Random Forest | 2.5267 | 3.0789 | 0.5429 | 0.6139 | 0.5043 |
| XGBoost | 2.6324 | 3.1706 | 0.5175 | 0.6054 | 0.4967 |
| Ridge | 2.7428 | 3.2505 | 0.4925 | 0.6076 | 0.4908 |

## Leitura

- Melhor score exploratorio em 2026: LightGBM (MAE medio 2.4567).
- Como ha apenas 7 corridas disponiveis, diferencas pequenas devem ser tratadas como indicio, nao conclusao definitiva.
- A comparacao 2026 serve para discutir drift/domain shift e preparar a Fase 2.

## Analise por faixa de grid

A analise por faixa de grid foi gerada em `reports/modelagem/analise_erro_2026_por_grid.csv`.

Nao foi gerada analise por tipo de circuito porque a base final 2026 nao possui uma coluna categorica confiavel de circuito urbano/permanente. Usar essa analise exigiria adicionar uma tabela manual de classificacao de circuitos.

## Artefatos

- `reports/modelagem/predicoes_2026_semana3.csv`
- `reports/modelagem/metricas_2026_por_corrida.csv`
- `reports/modelagem/metricas_2026_resumo.csv`
- `reports/modelagem/analise_erro_2026_por_grid.csv`
- `reports/modelagem/figures/semana3_2026/`
