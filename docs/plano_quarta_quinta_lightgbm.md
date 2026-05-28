# Plano de Implementacao - Quarta e Quinta da Semana 2

## Contexto

Este plano revisa as atividades de quarta e quinta da Semana 2 considerando a inclusao do LightGBM como terceiro algoritmo candidato. A arquitetura original estava centrada em XGBoost e Random Forest, mas o cronograma revisado adicionou LightGBM com base no resultado reportado por Barra et al. (2025).

Observacao de calendario: em 2026, quarta-feira corresponde a 27/05/2026 e quinta-feira corresponde a 28/05/2026. O cronograma em DOCX chama 26/05 de quarta e 27/05 de quinta, entao vale corrigir essa nomenclatura no documento final.

## Diagnostico do Cronograma Atual

O cronograma esta correto na direcao geral: usar a Semana 2 para rodar os tres algoritmos, comparar metricas walk-forward e escolher dois finalistas. A principal correcao recomendada e mudar a prioridade da quarta e da quinta.

Como os resultados preliminares sem tuning ja colocam o LightGBM levemente a frente dos demais modelos, ele nao deve ficar espremido no final da quinta-feira depois do Random Forest.

Resultados preliminares atuais:

| Modelo | MAE medio | MAE std | RMSE medio | R2 medio | Kendall tau medio | Top-3 medio |
|---|---:|---:|---:|---:|---:|---:|
| LightGBM | 2.3628 | 0.1137 | 3.0464 | 0.6511 | 0.6464 | 0.1995 |
| XGBoost | 2.4258 | 0.1779 | 3.1047 | 0.6367 | 0.6362 | 0.1427 |
| Random Forest | 2.4422 | 0.1192 | 3.1028 | 0.6382 | 0.6355 | 0.2134 |

Leitura: LightGBM lidera em MAE, RMSE, R2 e Kendall tau antes do tuning. Random Forest ainda e relevante por estabilidade, robustez e papel comparativo na arquitetura.

## Fundamentacao Bibliografica

As decisoes abaixo seguem a arquitetura proposta:

| Componente | Referencias da arquitetura | Uso no plano |
|---|---|---|
| Walk-forward validation e time-decay | Henderson et al. [9], Tan et al. [18] | Manter validacao temporal sem embaralhamento e pesos decrescentes por temporada. |
| Coeficientes RAPM | Henderson et al. [9], Snoeks [10] | Preservar `driver_coef_rapm` e `constructor_coef_rapm` como features centrais e rodar Ridge como baseline. |
| XGBoost | Chen e Guestrin [19], Barra et al. [3], Alonso et al. [4] | Manter como candidato principal de boosting. |
| Random Forest | Breiman [20], Ruan et al. [2] | Manter como modelo robusto e contraponto ao boosting. |
| LightGBM | Barra et al. [3] | Usar como terceiro candidato empirico na Fase 1. Recomenda-se adicionar a referencia original do LightGBM na bibliografia final. |
| Concept drift 2026 | Lu et al. [15], Thomas et al. [12], Chen et al. [11] | Escolher modelos finalistas pensando tambem na Fase 2 e no teste de degradacao em 2026. |
| Transfer learning e TrAdaBoost | Pan e Yang [13], Zhuang et al. [14], Dai et al. [21] | Selecionar os dois finalistas que serao reaproveitados na Fase 2 com adaptacao. |
| Metricas | Henderson et al. [9], Alonso et al. [4], Polishchuk [1] | Avaliar MAE, RMSE, R2, Kendall tau e top-3. |

## Plano Recomendado

## Quarta-feira - 27/05/2026

Objetivo: priorizar os modelos boosting, porque sao os candidatos mais fortes para dados tabulares e porque o LightGBM ja liderou os resultados preliminares.

1. Validar o recorte real da base
   - Confirmar que os arquivos de modelagem usam `2018-2025`, nao `2014-2025`.
   - Documentar no relatorio que a arquitetura previa 2014+, mas a base final disponivel da implementacao esta em `2018-2025`.
   - Manter folds:
     - treino `2018-2022` -> validacao `2023`;
     - treino `2018-2023` -> validacao `2024`;
     - treino `2018-2024` -> validacao `2025`.

2. Criar ou consolidar infraestrutura comum de tuning
   - Centralizar carregamento de dados.
   - Centralizar calculo de `sample_weight` com time-decay.
   - Centralizar avaliacao com `metricas.py`.
   - Garantir que todo modelo seja recriado a cada fold.

3. Rodar tuning Optuna do XGBoost
   - Trials recomendados: 50.
   - Parametros:
     - `n_estimators`: 100-500;
     - `max_depth`: 3-10;
     - `learning_rate`: 0.01-0.3;
     - `subsample`: 0.6-1.0;
     - `colsample_bytree`: 0.6-1.0;
     - `reg_alpha`: 0-1;
     - `reg_lambda`: 0-1.
   - Salvar:
     - melhores hiperparametros;
     - metricas fold a fold;
     - predicoes;
     - tempo de tuning;
     - dataframe de trials do Optuna.

4. Rodar tuning Optuna do LightGBM
   - Trials recomendados: 50.
   - Se houver restricao de tempo, usar 30 trials no LightGBM, mas somente se XGBoost e Random Forest mantiverem 50.
   - Parametros:
     - `n_estimators`: 100-500;
     - `max_depth`: 3-10;
     - `learning_rate`: 0.01-0.3;
     - `num_leaves`: 20-150;
     - `min_child_samples`: 5-50;
     - `subsample`: 0.6-1.0;
     - `colsample_bytree`: 0.6-1.0;
     - `reg_alpha`: 0-1;
     - `reg_lambda`: 0-1.
   - Usar `sample_weight` em `LGBMRegressor.fit(...)`.
   - Evitar reutilizar objetos `lgb.Dataset` entre folds.

Entregaveis da quarta:

- `src/tuning_xgboost.py`
- `src/tuning_lightgbm.py`
- `reports/modelagem/tuning_xgboost_trials.csv`
- `reports/modelagem/tuning_lightgbm_trials.csv`
- `reports/modelagem/metricas_tuned_xgboost.csv`
- `reports/modelagem/metricas_tuned_lightgbm.csv`
- `reports/modelagem/predicoes_tuned_xgboost.csv`
- `reports/modelagem/predicoes_tuned_lightgbm.csv`
- `reports/modelagem/melhores_parametros_xgboost.json`
- `reports/modelagem/melhores_parametros_lightgbm.json`

## Quinta-feira - 28/05/2026

Objetivo: completar a comparacao com Random Forest, rodar o baseline Ridge e consolidar uma decisao preliminar dos finalistas.

1. Rodar tuning Optuna do Random Forest
   - Trials recomendados: 50.
   - Parametros:
     - `n_estimators`: 100-500;
     - `max_depth`: 3-15;
     - `max_features`: `sqrt`, `log2`, `0.5`;
     - `min_samples_split`: 2-10;
     - `min_samples_leaf`: 1-5.
   - Salvar os mesmos artefatos dos modelos boosting.

2. Rodar Ridge Regression baseline
   - Usar Ridge com time-decay.
   - Varrer `alpha` em escala logaritmica, por exemplo `0.01` a `100`.
   - Avaliar por walk-forward nos mesmos folds.
   - Justificativa: o baseline linear e essencial porque a arquitetura usa RAPM/Ridge como fundamento metodologico dos coeficientes de piloto e construtor.

3. Consolidar tabela final dos quatro modelos
   - Modelos:
     - XGBoost;
     - Random Forest;
     - LightGBM;
     - Ridge.
   - Colunas obrigatorias:
     - MAE medio;
     - MAE desvio padrao;
     - RMSE medio;
     - R2 medio;
     - Kendall tau medio;
     - top-3 medio;
     - melhor fold;
     - pior fold;
     - tempo de tuning.

4. Fazer decisao preliminar dos dois finalistas
   - A decisao ainda pode ser confirmada no fim de semana apos feature selection.
   - Criterios:
     - menor MAE medio;
     - maior Kendall tau;
     - menor desvio padrao entre folds;
     - custo computacional;
     - coerencia da feature importance com a literatura.

5. Registrar decisao metodologica
   - LightGBM deve ser descrito como candidato empirico adicional, nao como substituicao automatica do XGBoost.
   - O texto deve deixar claro que tres algoritmos foram avaliados e dois serao selecionados com base nos resultados.

Entregaveis da quinta:

- `src/tuning_random_forest.py`
- `src/otimizacao_ridge_lambda.py`
- `reports/modelagem/tuning_random_forest_trials.csv`
- `reports/modelagem/metricas_tuned_random_forest.csv`
- `reports/modelagem/metricas_ridge_baseline.csv`
- `reports/modelagem/tabela_metricas_tuned_4modelos.csv`
- `reports/modelagem/tabela_metricas_tuned_4modelos_resumo.csv`
- `reports/modelagem/decisao_preliminar_algoritmos.md`

## Plano de Implementacao em Codigo

1. Criar `src/tuning_common.py`
   - `carregar_dados_modelagem()`
   - `calcular_sample_weight()`
   - `iterar_folds_walk_forward()`
   - `avaliar_modelo_walk_forward()`
   - `salvar_resultados_tuning()`

2. Adaptar cada script de tuning para usar `tuning_common.py`
   - Evita divergencia entre XGBoost, LightGBM e Random Forest.
   - Garante que os tres modelos usem exatamente os mesmos folds, metricas e pesos.

3. Definir objetivo Optuna
   - Objetivo principal: minimizar MAE medio.
   - Criterio secundario para desempate: maior Kendall tau medio.
   - Registrar o tempo total de execucao.

4. Rodar scripts em ordem
   - `python src/tuning_xgboost.py`
   - `python src/tuning_lightgbm.py`
   - `python src/tuning_random_forest.py`
   - `python src/otimizacao_ridge_lambda.py`
   - `python src/consolidar_metricas_tuned.py`

5. Conferir qualidade dos resultados
   - MAE alvo: menor ou igual a 2.5.
   - RMSE alvo: menor ou igual a 3.0.
   - R2 alvo: maior ou igual a 0.75.
   - Kendall tau alvo: maior ou igual a 0.60.
   - Top-3 alvo: maior ou igual a 70%, mas interpretar com cuidado porque a metrica atual e estrita.

## Criterio de Decisao dos Finalistas

| Cenario | Decisao recomendada |
|---|---|
| LightGBM vence ou empata com XGBoost e tem custo menor | LightGBM vira finalista. |
| XGBoost vence LightGBM com diferenca clara de MAE/Kendall tau | XGBoost permanece finalista. |
| Random Forest perde pouco, mas tem menor variancia | Manter Random Forest como segundo finalista pode ser defensavel. |
| Random Forest fica claramente atras dos dois boosting | Finalistas devem ser XGBoost e LightGBM. |
| Ridge chega perto dos ensembles | Discutir como resultado relevante, mas manter Ridge como baseline, nao finalista principal. |

## Ajuste Recomendado na Arquitetura

A arquitetura deve ser atualizada para mencionar LightGBM na Fase 1 como terceiro candidato experimental. A estrutura mais defensavel e:

> Foram avaliados tres algoritmos de aprendizado supervisionado para regressao da posicao final: XGBoost, Random Forest e LightGBM. XGBoost e Random Forest compoem a arquitetura original por representarem, respectivamente, boosting regularizado e bagging robusto. LightGBM foi incluido como terceiro candidato empirico devido ao desempenho reportado por Barra et al. (2025) em dados tabulares de Formula 1. A selecao dos dois modelos finalistas foi feita por walk-forward validation, considerando MAE, Kendall tau, estabilidade entre folds e custo computacional.

Recomenda-se tambem adicionar a referencia original do LightGBM na bibliografia tecnica final:

Ke, G. et al. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. Advances in Neural Information Processing Systems, 2017.

## Conclusao

O cronograma esta bom, mas deve ser reordenado: quarta-feira deve focar em XGBoost e LightGBM; quinta-feira deve focar em Random Forest, Ridge baseline e consolidacao comparativa. Essa organizacao respeita melhor os resultados preliminares, mantem coerencia com a arquitetura e deixa a decisao dos finalistas mais defensavel na banca.
