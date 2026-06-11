# 11 — Plano de Estudos de Ablação

## Contexto

Este documento foi revisado após a remoção da métrica top-3 do pipeline principal. A avaliação oficial passa a usar apenas métricas comparáveis aos trabalhos de regressão/ranking revisados:

- MAE;
- RMSE;
- R²;
- Kendall τ.

Top-3 e classificadores de pódio deixam de ser metas deste TCC. Eles podem aparecer em relatórios históricos de experimentos, mas não fazem parte do critério oficial de seleção, tuning ou defesa.

---

## Motivação

Os resultados finais da Fase 1 mostram que:

| Modelo | MAE | RMSE | R² | Kendall τ | Score |
|---|---:|---:|---:|---:|---:|
| Ridge baseline | 2.2723 | 2.9574 | 0.6710 | 0.6543 | 0.5314 |
| LightGBM | 2.3172 | 3.0121 | 0.6587 | 0.6536 | 0.5279 |
| Random Forest | 2.3263 | 3.0121 | 0.6589 | 0.6503 | 0.5272 |
| XGBoost | 2.3415 | 3.0161 | 0.6578 | 0.6525 | 0.5269 |

O principal desafio restante é reduzir RMSE para abaixo de 3.0 nas árvores e aproximar R² de 0.67 sem sacrificar Kendall τ.

---

## Score Oficial

| Métrica | Peso |
|---|---:|
| MAE invertido | 0.35 |
| RMSE invertido | 0.20 |
| R² | 0.20 |
| Kendall τ | 0.25 |

Esse score é usado para tuning, comparação de modelos e ablações oficiais.

---

## Estudos Prioritários

### 1. Target fixo

Decisão metodológica: o alvo oficial do trabalho é sempre `finish_position`.

Essa decisão evita alterar a pergunta estatística do TCC. O modelo deve prever diretamente a posição final, mantendo comparabilidade entre algoritmos, ablações e trabalhos relacionados. Transformações como `delta_grid`, `rank_norm_grid20` ou `log1p_finish` podem ser registradas apenas como exploração histórica, mas não substituem o target oficial nem entram na decisão final.

Critério: todo experimento oficial deve manter `target_mode=finish`.

#### Resultado pareado LightGBM vs XGBoost

Em 11/06/2026 foi executada uma ablação pareada específica para LightGBM e XGBoost. A primeira bateria explorou transformações de target e indicou ganhos com `delta_grid`, mas essa alternativa foi rejeitada por mudar o alvo do problema. O estudo oficial foi então restringido a `target_mode=finish`, mantendo as mesmas configurações experimentais para ambos:

- mesmos folds de tuning: 2023 e 2024;
- mesmos folds de avaliação: 2023, 2024 e 2025;
- mesmo target oficial: `finish`;
- mesmos fatores de time-decay: `0.95` e `0.99`;
- mesmos perfis de score: `atual`, `rmse_r2`, `erro_continuo`;
- mesmo orçamento de busca: 20 trials por combinação e modelo.

Com target fixo, o melhor resultado continua próximo da linha oficial de modelagem:

| Modelo | Target | Decay | MAE | RMSE | R² | Kendall τ | Score |
|---|---|---:|---:|---:|---:|---:|---:|
| LightGBM | `finish` | 0.99 | 2.3226 | 3.0075 | 0.6599 | 0.6528 | 0.5278 |

O resultado com alvo fixo não atingiu simultaneamente RMSE < 3.0 e R² ≥ 0.66 nas árvores. Portanto, o caminho para melhorar as metas deve focar em hiperparâmetros, features, regularização, ensembles e validações de robustez, sem alterar o target.

Artefatos:

- `src/ablacao_pareada_lgbm_xgboost.py`;
- `reports/ablacao/pareada_lgbm_xgboost/resultados_pareados.csv`;
- `reports/ablacao/pareada_lgbm_xgboost/relatorio_pareado.md`.

### 2. Ensembles com Ridge

Ridge é o melhor baseline oficial, enquanto árvores fornecem interpretação não linear. Testar ensembles ponderados pode combinar os dois sinais.

| Ensemble | Objetivo |
|---|---|
| Ridge + LightGBM | Reduzir RMSE mantendo interpretabilidade de árvore |
| Ridge + Random Forest | Aumentar estabilidade em folds temporais |
| Ridge + LightGBM + Random Forest | Testar equilíbrio linear/não linear |

Critério: RMSE ≤ 3.0 e R² ≥ 0.66.

### 3. Ablação de features

Avaliar remoções controladas, sempre usando o score oficial sem top-3.

| Feature/Grupo | Hipótese |
|---|---|
| `avg_pit_stops_circuit` | Pode ser ruído fraco dependendo do fold |
| `altitude_m` | Baixa importância média |
| `season_factor` | Pode ser redundante com walk-forward/time-decay |
| `constructor_dnf_rate` | Baixo ganho, mas interpretável |

Critério: remover apenas se melhorar score composto e não prejudicar a justificativa metodológica.

### 4. Time-decay fino

O decay oficial é `0.99`. Testar `0.97`, `0.98`, `0.99` e `1.00` com retuning completo para confirmar robustez.

Critério: se a diferença for marginal, manter `0.99` por já estar documentado e validado.

---

## O Que Saiu do Plano

Foram removidos do escopo oficial:

- meta top-3 ≥ 70%;
- score com peso para top-3;
- classificador específico de pódio;
- ranking de experimentos por top-3.

Justificativa: top-3 é uma tarefa de classificação de pódio, não uma métrica comparável aos trabalhos de regressão causal de `finish_position`.

---

## Critério de Decisão

Um experimento só deve substituir o pipeline oficial se:

1. melhorar o score composto oficial;
2. não introduzir leakage temporal ou pós-corrida;
3. manter Kendall τ ≥ 0.60;
4. preservar uma narrativa metodológica defensável para a banca.
