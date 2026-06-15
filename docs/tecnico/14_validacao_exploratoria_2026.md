# 14 - Validacao Exploratoria em 2026

## Contexto

Esta etapa documenta o ultimo passo da Fase 1: aplicar os modelos treinados com dados historicos de 2018-2025 nas corridas de 2026 ja disponiveis na OpenF1.

O objetivo nao e substituir a avaliacao oficial do projeto. As metricas oficiais continuam sendo os folds walk-forward de 2023, 2024 e 2025. A avaliacao de 2026 funciona como um teste exploratorio de mudanca temporal, isto e, uma primeira verificacao de como os modelos se comportam quando expostos a uma temporada posterior ao periodo usado no treinamento.

Essa analise e importante porque o projeto trata mudancas regulatorias como possivel problema de `domain shift`: a relacao entre historico de piloto/construtor, classificacao, forma recente e resultado final pode mudar em uma nova era tecnica.

---

## Dados disponiveis

A base 2026 foi gerada por `src/update_openf1_2026.py`, que consulta a OpenF1, atualiza os arquivos raw de 2026 e gera:

- `data/processed/openf1_2026_available.csv`;
- `data/processed/relatorio_update_2026.txt`.

Na execucao atual, a OpenF1 retornou resultados para 7 corridas:

| Round | Corrida |
|---:|---|
| 1 | Australian Grand Prix |
| 2 | Chinese Grand Prix |
| 3 | Japanese Grand Prix |
| 6 | Miami Grand Prix |
| 7 | Canadian Grand Prix |
| 8 | Monaco Grand Prix |
| 9 | Barcelona Grand Prix |

Bahrain e Saudi Arabian Grand Prix aparecem no calendario, mas a API nao retornou resultado de `session_result` na atualizacao executada. Por isso, essas etapas nao entraram na avaliacao.

Resumo da base processada:

| Item | Valor |
|---|---:|
| Corridas processadas | 7 |
| Linhas totais | 123 |
| Pilotos unicos | 22 |
| Linhas com `finish_position` ausente | 9 |
| Linhas validas para avaliacao | 114 |

---

## Validacao de schema

A avaliacao 2026 usa exatamente as 13 features finais do dataset de modelagem 2018-2025:

| Feature | Status |
|---|---|
| `qualifying_position` | OK |
| `constructor_coef_rapm` | OK |
| `recent_form_5` | OK |
| `driver_constructor_synergy` | OK |
| `constructor_wins_total` | OK |
| `driver_coef_rapm` | OK |
| `track_complexity` | OK |
| `tire_compound_start` | OK |
| `season_factor` | OK |
| `avg_pit_stops_circuit` | OK |
| `constructor_dnf_rate` | OK |
| `grid_penalty` | OK |
| `altitude_m` | OK |

Nao houve features ausentes nem valores nulos nas 13 features finais. As 9 linhas com `finish_position` ausente foram removidas antes do calculo das metricas, pois o alvo real e obrigatorio para avaliar erro.

---

## Protocolo experimental

Os quatro modelos foram treinados novamente usando todo o periodo historico disponivel ate 2025:

- Ridge baseline;
- LightGBM;
- Random Forest;
- XGBoost.

O protocolo manteve as mesmas decisoes da modelagem oficial:

- mesmo target: `finish_position`;
- mesmas 13 features finais;
- mesmos hiperparametros salvos nos artefatos de modelagem;
- mesmo fator de time-decay escolhido;
- mesmo conjunto de metricas: MAE, RMSE, R2 e Kendall tau.

A diferenca em relacao ao fold oficial de 2025 e que 2026 nao faz parte do pipeline historico consolidado. A base 2026 e montada a partir dos dados disponiveis na OpenF1 e usa snapshots historicos de fim de 2025 para features de piloto, construtor e forma recente. Portanto, o resultado deve ser tratado como validacao exploratoria, nao como avaliacao final definitiva.

Scripts principais:

- `src/update_openf1_2026.py`;
- `src/avaliar_2026_semana3.py`.

---

## Resultados

Fonte: `reports/modelagem/metricas_2026_resumo.csv`.

| Modelo | MAE medio | RMSE medio | R2 medio | Kendall tau medio | Score exploratorio |
|---|---:|---:|---:|---:|---:|
| LightGBM | 2.4567 | 3.0059 | 0.5614 | 0.6390 | 0.5122 |
| Random Forest | 2.5267 | 3.0789 | 0.5429 | 0.6139 | 0.5043 |
| XGBoost | 2.6324 | 3.1706 | 0.5175 | 0.6054 | 0.4967 |
| Ridge | 2.7428 | 3.2505 | 0.4925 | 0.6076 | 0.4908 |

O LightGBM teve o melhor desempenho exploratorio em 2026, liderando MAE, RMSE, R2, Kendall tau e score composto. O Random Forest ficou em segundo, seguido por XGBoost e Ridge.

---

## Comparacao com os folds oficiais

No resultado oficial 2023-2025, o Ridge foi o melhor modelo global, enquanto LightGBM foi a melhor arvore:

| Modelo | MAE oficial 2023-2025 | MAE 2026 exploratorio | Diferenca |
|---|---:|---:|---:|
| Ridge | 2.2723 | 2.7428 | +0.4705 |
| LightGBM | 2.3172 | 2.4567 | +0.1395 |
| Random Forest | 2.3263 | 2.5267 | +0.2004 |
| XGBoost | 2.3415 | 2.6324 | +0.2909 |

Leitura:

- todos os modelos pioraram em MAE no teste 2026;
- a queda foi mais forte no Ridge;
- LightGBM apresentou a menor degradacao em MAE;
- o ranking exploratorio de 2026 difere do ranking oficial 2023-2025.

Essa diferenca e coerente com a hipotese de drift temporal: modelos que capturam relacoes historicas estaveis podem perder desempenho quando a distribuicao de dados muda.

---

## Interpretacao

O resultado de 2026 sugere tres pontos importantes.

Primeiro, houve degradacao de desempenho. O R2 caiu de aproximadamente 0.66-0.67 nos folds oficiais para 0.49-0.56 no teste 2026, e o RMSE ficou acima de 3.0 para todos os modelos exceto por arredondamento muito proximo no LightGBM. Isso indica que a variancia dos resultados de 2026 foi mais dificil de explicar com as features historicas disponiveis.

Segundo, o LightGBM foi mais robusto no cenario exploratorio. Embora Ridge tenha vencido no protocolo oficial 2023-2025, o LightGBM degradou menos em 2026. Isso pode indicar que, em um contexto de mudanca temporal, as arvores de boosting conseguem explorar interacoes residuais entre qualifying, forma recente e features de construtor de forma mais adaptavel que o baseline linear.

Terceiro, a feature `qualifying_position` continua sendo um sinal muito forte. Como a avaliacao usa a classificacao real de 2026, parte da mudanca de performance do carro ja esta refletida nessa feature. Isso reduz o tamanho aparente do drift, pois o modelo recebe uma informacao atualizada do fim de semana da corrida. Mesmo assim, a queda em R2 e RMSE mostra que o restante da estrutura historica ficou menos explicativo.

---

## Limitacoes

Esta validacao tem limitacoes metodologicas importantes:

1. A amostra ainda e pequena: apenas 7 corridas e 114 linhas validas.
2. A OpenF1 nao retornou resultado para todas as corridas ja passadas no calendario.
3. Algumas features de piloto/construtor em 2026 usam snapshot de fim de 2025, nao atualizacao corrida a corrida.
4. As linhas com `finish_position` ausente foram removidas, o que torna a avaliacao condicional aos resultados disponiveis.
5. A analise usa `qualifying_position` real de 2026, uma feature pre-corrida forte que absorve parte da mudanca de performance da nova temporada.

Por esses motivos, os resultados devem ser apresentados como evidencia inicial de drift e motivacao para a Fase 2, nao como conclusao definitiva sobre toda a temporada 2026.

---

## Artefatos gerados

| Artefato | Descricao |
|---|---|
| `data/processed/openf1_2026_available.csv` | Base 2026 processada com features finais e target |
| `data/processed/relatorio_update_2026.txt` | Relatorio de cobertura da atualizacao 2026 |
| `reports/modelagem/metricas_2026_resumo.csv` | Metricas medias por modelo |
| `reports/modelagem/metricas_2026_por_corrida.csv` | Metricas por corrida e modelo |
| `reports/modelagem/predicoes_2026_semana3.csv` | Predicoes individuais dos modelos em 2026 |
| `reports/modelagem/analise_erro_2026_por_grid.csv` | Erro por faixa de grid |
| `reports/modelagem/analise_2026_semana3.md` | Relatorio resumido da avaliacao exploratoria |
| `reports/modelagem/validacao_2026_semana3.md` | Validacao de schema, ausencias e resultados |
| `reports/modelagem/figures/semana3_2026/` | Graficos da avaliacao 2026 |

---

## Conclusao

A validacao exploratoria em 2026 confirma que o pipeline consegue ser aplicado a dados futuros mantendo o mesmo contrato de features e metricas. Com as 7 corridas disponiveis na OpenF1, houve queda clara de desempenho em relacao aos folds oficiais 2023-2025, especialmente em R2 e RMSE.

Esse comportamento sustenta a motivacao metodologica da Fase 2: investigar estrategias de adaptacao temporal, como time-decay refinado, janelas recentes, retreinamento incremental e metodos de transfer learning para reduzir a degradacao causada por mudanca de dominio.
