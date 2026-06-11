# Resumo das mudanças para subir no GitHub

Data: 11/06/2026

Este arquivo resume a revisão recente do repositório para facilitar a leitura da equipe.

## Mudanças metodológicas principais

- O critério de seleção deixou de ser apenas MAE médio e passou a ser um score composto multi-métrica: MAE, RMSE, R² e Kendall tau.
- O time-decay do walk-forward foi revisado: o valor final documentado é `0.99`, escolhido por score composto nos folds 2023-2024.
- A seleção de features foi refeita com RFE temporal multi-fold em 2023, 2024 e 2025.
- O X final passou de 15 para 13 features, com 2.943 linhas e 0 NaN.
- Os finalistas de árvore foram revisados para LightGBM e Random Forest. XGBoost fica documentado como terceiro modelo de árvore avaliado.
- A métrica top-3 foi removida do pipeline oficial por não ser comparável à regressão causal de `finish_position`.
- Estudos exploratórios com transformação do target foram marcados como históricos/obsoletos; o target oficial permanece `finish_position`.
- Foram adicionados estudos de ablação em `reports/ablacao/` e scripts correspondentes em `src/`.

## Contrato atual de features

O contrato canônico está em `models/feature_selection/features_modelagem_2018_2025.json`.

Features finais:

```text
qualifying_position
constructor_coef_rapm
recent_form_5
driver_constructor_synergy
constructor_wins_total
driver_coef_rapm
track_complexity
tire_compound_start
season_factor
avg_pit_stops_circuit
constructor_dnf_rate
grid_penalty
altitude_m
```

## Arquivos principais atualizados

- `docs/tecnico/07_selecao_features.md`
- `docs/tecnico/08_walk_forward_time_decay.md`
- `docs/tecnico/09_modelagem_tuning.md`
- `docs/tecnico/10_resultados_feature_importance.md`
- `docs/tecnico/11_plano_estudos_ablacao.md`
- `docs/tecnico/12_baselines_literatura.md`
- `docs/tecnico/13_material_complementar_discussao_modelos.md`
- `reports/modelagem/decisao_algoritmos.md`
- `reports/ablacao/relatorio_estudos_ablacao_completo.md`

## Checks feitos antes do push

- `python3 -m py_compile src/*.py`
- Validação de JSONs do repositório
- Checagem de shapes e NaN nos CSVs principais
- `git diff --check`
- Varredura de consistência para referências antigas a top-3, target transformado e finalistas desatualizados

## Atenções restantes

- `openf1_2026_available.csv` é artefato de Semana 3 e deve ser realinhado ao schema final de 13 features antes de nova rodada de drift.
- Ainda existem documentos históricos que citam estados antigos como contexto de auditoria. Para decisões atuais, priorizar os documentos técnicos 07 a 12 e este resumo.
