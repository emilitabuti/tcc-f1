# Validacao Semana 3 - Dados 2026

## Status

`data/processed/openf1_2026_available.csv` existe com 123 linhas, 22 pilotos unicos e 7 corridas.

Leitura metodologica: usar 2026 como analise exploratoria de mudanca temporal, nao como resultado principal da Fase 1.

## Cobertura de features

- `qualifying_position`: OK
- `constructor_coef_rapm`: OK
- `recent_form_5`: OK
- `driver_constructor_synergy`: OK
- `constructor_wins_total`: OK
- `driver_coef_rapm`: OK
- `track_complexity`: OK
- `tire_compound_start`: OK
- `season_factor`: OK
- `avg_pit_stops_circuit`: OK
- `constructor_dnf_rate`: OK
- `grid_penalty`: OK
- `altitude_m`: OK

## Ausencias

- Features finais ausentes no schema: nenhuma.
- Valores nulos nas 13 features finais: nenhum.
- `finish_position` ausente: 9 linhas.
- Linhas validas para avaliacao dos modelos: 114.

## Relatorio de origem

- `data/processed/relatorio_update_2026.txt`

## Avaliacao nos modelos

A avaliacao exploratoria foi reexecutada em `src/avaliar_2026_semana3.py` apos atualizar a base 2026 para 7 corridas.

Artefatos:

- `reports/modelagem/analise_2026_semana3.md`
- `reports/modelagem/metricas_2026_resumo.csv`
- `reports/modelagem/metricas_2026_por_corrida.csv`
- `reports/modelagem/predicoes_2026_semana3.csv`
- `reports/modelagem/analise_erro_2026_por_grid.csv`
- `reports/modelagem/figures/semana3_2026/`

Resumo exploratorio:

| Modelo | MAE medio | RMSE medio | R2 medio | Kendall tau medio | Score exploratorio |
|---|---:|---:|---:|---:|---:|
| LightGBM | 2.4567 | 3.0059 | 0.5614 | 0.6390 | 0.5122 |
| Random Forest | 2.5267 | 3.0789 | 0.5429 | 0.6139 | 0.5043 |
| XGBoost | 2.6324 | 3.1706 | 0.5175 | 0.6054 | 0.4967 |
| Ridge | 2.7428 | 3.2505 | 0.4925 | 0.6076 | 0.4908 |

Leitura: com 7 corridas, o desempenho caiu em relacao ao teste parcial com 4 corridas, o que e coerente com a hipotese de drift/domain shift. LightGBM segue como melhor modelo exploratorio em 2026.
