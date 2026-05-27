# Plano detalhado - Segunda da Semana 2: Modelagem

Data de referencia do cronograma: segunda-feira, Semana 2.

Este plano revisa se o cronograma da segunda-feira esta adequado em relacao a
`ArquiteturaProposta.pdf` e detalha a implementacao que deve ser seguida. O foco e
concluir a infraestrutura de modelagem temporal antes de entrar em tuning, Random Forest,
Ridge baseline e RFE final.

---

## Veredito sobre o cronograma

O cronograma esta bom e a ordem das tarefas esta correta:

1. validar schema e fold de 2025;
2. implementar metricas;
3. otimizar time-decay;
4. rodar walk-forward;
5. aplicar `sample_weight` no XGBoost.

Essa ordem e coerente com a arquitetura porque a modelagem nao pode ser embaralhada: a
validacao precisa respeitar a ordem temporal, e o time-decay precisa ser escolhido em
folds anteriores antes de avaliar 2025.

O principal ajuste necessario e metodologico: a arquitetura e o cronograma falam em
treino desde 2014, mas a base final de modelagem do projeto esta em `2018-2025`.
Portanto, os folds reais devem ser:

| Cronograma original | Execucao real no projeto |
|---|---|
| treino 2014-2022 -> validacao 2023 | treino 2018-2022 -> validacao 2023 |
| treino 2014-2023 -> validacao 2024 | treino 2018-2023 -> validacao 2024 |
| treino 2014-2024 -> validacao 2025 | treino 2018-2024 -> validacao 2025 |

Esse ajuste deve ser documentado no relatorio, porque afeta a comparacao direta com
benchmarks que usam toda a era hibrida desde 2014.

---

## Fundamentacao bibliografica usada

| Decisao da segunda-feira | Referencias da arquitetura |
|---|---|
| Walk-forward validation temporal | Henderson et al. [9] |
| Coeficientes RAPM e modelo linear com decomposicao piloto/construtor | Henderson et al. [9], Snoeks [10] |
| Time-decay e pesos menores para dados antigos | Henderson et al. [9], Tan et al. [18] |
| XGBoost como primeiro modelo tabular forte | Chen e Guestrin [19], Barra et al. [3], Alonso et al. [4] |
| Random Forest como comparacao posterior | Breiman [20], Ruan et al. [2] |
| Metricas MAE, RMSE, R2 e Kendall tau | Henderson et al. [9], Alonso et al. [4] |
| Acuracia de top-3/podio como metrica complementar | Polishchuk [1] |
| Evitar leakage de features pos-corrida | Ruan et al. [2], Barra et al. [3], Koopman et al. [5] |
| Drift regulatorio futuro em 2026 | Lu et al. [15], Chen et al. [11], Thomas et al. [12] |

---

## Escopo da segunda-feira

### Dentro do escopo

- Validar se `X` e `y` de modelagem estao alinhados.
- Garantir que 2025 esta integrado como fold temporal final.
- Implementar/validar `src/metricas.py`.
- Implementar/validar `src/otimizacao_time_decay.py`.
- Implementar/validar `src/walk_forward.py`.
- Escolher o fator de time-decay usando apenas 2023 e 2024.
- Rodar XGBoost sem tuning Optuna com `sample_weight`.
- Gerar relatorio da segunda-feira.

### Fora do escopo

- Random Forest.
- Tuning Optuna.
- Ridge baseline.
- RFE final.
- SHAP.
- Analise definitiva de drift 2026.

Essas tarefas entram depois porque dependem da infraestrutura temporal validada nesta
segunda-feira.

---

## Estado atual do projeto

Ja existem os arquivos principais da segunda-feira:

- `src/metricas.py`
- `src/otimizacao_time_decay.py`
- `src/walk_forward.py`
- `src/validar_schema_2025_modelagem.py`

Ja existem bases prontas:

- `data/processed/dataset_modelagem_X_2018_2025.csv`
- `data/processed/dataset_modelagem_y_2018_2025.csv`
- `data/processed/dataset_modelagem_2018_2025.csv`
- `data/processed/openf1_2025_clean.csv`
- `data/processed/validacao_2025_clean.csv`

Ja existem artefatos gerados em `reports/modelagem/`:

- `otimizacao_time_decay_xgboost.csv`
- `otimizacao_time_decay_xgboost_resumo.csv`
- `time_decay_escolhido_xgboost.txt`
- `predicoes_walk_forward_xgboost.csv`
- `metricas_walk_forward_xgboost.csv`
- `relatorio_segunda_semana2_xgboost.txt`
- `validacao_schema_2025_modelagem.txt`

Portanto, o trabalho agora nao e "comecar do zero"; e revisar, executar, validar e
documentar.

---

## Plano de implementacao detalhado

### 1. Conferir schema e alinhamento de dados

Objetivo: garantir que o fold de 2025 tem exatamente as features esperadas e que `X` e
`y` continuam alinhados.

Comandos:

```bash
python src/validar_schema_2025_modelagem.py
python src/selecao_features_modelagem.py
```

Checklist:

- `dataset_modelagem_X_2018_2025.csv` nao pode conter `finish_position`.
- `dataset_modelagem_y_2018_2025.csv` deve conter `season`, `round`, `driver_id`,
  `finish_position`.
- `X` e `y` devem ter o mesmo numero de linhas.
- Nenhuma coluna proibida por leakage pode entrar em `X`.
- A temporada 2025 deve existir em `y`.

Criterio de aceite:

- O script de schema termina sem erro.
- O relatorio registra compatibilidade do schema para walk-forward 2025.

Base bibliografica: a validacao anti-leakage segue Ruan et al. [2], Barra et al. [3] e
Koopman et al. [5], porque variaveis pos-corrida inflariam artificialmente a performance.

---

### 2. Revisar `src/metricas.py`

Objetivo: manter as metricas independentes do modelo, reutilizaveis para XGBoost,
Random Forest e Ridge.

Metricas obrigatorias:

- `mae`
- `rmse`
- `r2`
- `kendall_tau`
- `top3_accuracy`

Checklist tecnico:

- `calcular_metricas(df_pred)` deve exigir as colunas:
  - `season`
  - `round`
  - `driver_id`
  - `finish_position`
  - `pred_finish_position`
- Kendall tau deve ser calculado por corrida e depois agregado por media.
- Top-3 deve comparar o conjunto dos tres primeiros reais contra o conjunto dos tres
  primeiros previstos.

Observacao metodologica:

A `top3_accuracy` atual e uma metrica estrita: ela so conta acerto quando o conjunto
inteiro do podio previsto bate com o conjunto real. Por isso ela tende a ficar bem menor
que a referencia de Polishchuk [1], que usa podio como benchmark de acuracia, mas pode
ter formulacao diferente. No relatorio, chamar essa metrica de "acuracia top-3 estrita"
evita comparacao injusta.

Base bibliografica: MAE e Kendall tau sao coerentes com Henderson et al. [9]; top-3 vem
como comparacao complementar inspirada em Polishchuk [1].

---

### 3. Otimizar o fator de time-decay

Objetivo: escolher o fator de decaimento usando somente folds anteriores ao teste 2025.

Comando:

```bash
python src/otimizacao_time_decay.py
```

Fatores testados:

- `0.50`
- `0.65`
- `0.75`
- `0.85`
- `0.95`

Folds usados na escolha:

- treino 2018-2022 -> validacao 2023
- treino 2018-2023 -> validacao 2024

Regra de decisao:

- escolher o fator com menor MAE medio em 2023-2024;
- usar Kendall tau e top-3 apenas como metricas auxiliares;
- nao usar 2025 para escolher o fator, porque 2025 deve funcionar como validacao final.

Resultado atual observado:

| Decay | MAE medio 2023-2024 |
|---|---:|
| 0.85 | 2.414576 |
| 0.75 | 2.414762 |
| 0.65 | 2.415667 |
| 0.95 | 2.426356 |
| 0.50 | 2.435438 |

Decisao atual: `0.85`.

Observacao importante: `0.85` ganhou de `0.75` por margem muito pequena. Como `0.75`
e o ponto de partida sugerido pelo RAPM, vale documentar que a escolha por `0.85` foi
empirica, mas que os dois fatores sao praticamente equivalentes nesta amostra.

Base bibliografica: Henderson et al. [9] fundamenta o uso de time-decay no contexto
RAPM/F1; Tan et al. [18] fundamenta a ideia geral de pesos temporais decrescentes em
aprendizado nao-estacionario.

---

### 4. Rodar walk-forward com XGBoost e sample weights

Objetivo: validar a infraestrutura completa da segunda-feira com XGBoost ainda sem tuning.

Comando:

```bash
python src/walk_forward.py
```

Folds finais:

- treino 2018-2022 -> validacao 2023
- treino 2018-2023 -> validacao 2024
- treino 2018-2024 -> validacao 2025

Peso temporal esperado:

```text
peso = decay ^ (valid_season - season)
```

Com `decay = 0.85`, dados da temporada imediatamente anterior ao fold recebem peso maior
que dados antigos. Isso preserva o principio da arquitetura: corridas mais recentes devem
ter mais influencia porque refletem melhor o estado competitivo atual.

Resultados atuais:

| Validacao | MAE | RMSE | R2 | Kendall tau | Top-3 estrito |
|---|---:|---:|---:|---:|---:|
| 2023 | 2.564548 | 3.238925 | 0.594347 | 0.625833 | 0.136364 |
| 2024 | 2.264604 | 2.943094 | 0.683637 | 0.669066 | 0.166667 |
| 2025 | 2.469857 | 3.191213 | 0.619424 | 0.613309 | 0.125000 |

Leitura rapida:

- O MAE 2025 de `2.469857` fica dentro da meta da arquitetura (`<= 2.5`).
- O Kendall tau 2025 de `0.613309` fica acima da meta (`>= 0.60`) e proximo da
  referencia RAPM (`0.625`).
- O RMSE 2025 de `3.191213` ainda fica acima da meta (`<= 3.0`), ponto para atacar no
  tuning Optuna.
- A top-3 estrita esta baixa; nao tratar isso como falha central antes de revisar a
  definicao da metrica.

Base bibliografica: XGBoost e justificado por Chen e Guestrin [19] e pelos trabalhos de
predicao em F1 que usam modelos tabulares fortes [3], [4].

---

### 5. Gerar relatorio da segunda-feira

Objetivo: deixar um rastro auditavel para a metodologia.

Arquivo:

- `reports/modelagem/relatorio_segunda_semana2_xgboost.txt`

Conteudo minimo:

- bases usadas;
- ajuste do recorte 2014 -> 2018;
- folds usados;
- fator de time-decay testado e escolhido;
- metricas por fold;
- observacao de que o modelo ainda esta sem tuning;
- observacao de que Random Forest ficou para etapa posterior.

Checklist de texto metodologico:

- explicar que 2025 nao foi usado para escolher o decay;
- explicar que `0.85` foi escolhido por MAE medio em 2023-2024;
- citar que `0.75` veio da referencia RAPM, mas foi tratado como ponto de partida;
- registrar que a comparacao com benchmarks 2014+ deve considerar o recorte menor
  2018-2025.

---

## Ordem recomendada para executar na pratica

1. Rodar validacao de schema.
2. Rodar selecao/congelamento de features se houver qualquer mudanca em FE.
3. Rodar otimizacao de time-decay.
4. Conferir `reports/modelagem/time_decay_escolhido_xgboost.txt`.
5. Rodar walk-forward XGBoost.
6. Conferir `reports/modelagem/metricas_walk_forward_xgboost.csv`.
7. Atualizar o relatorio da segunda-feira com os resultados.
8. Registrar no texto da metodologia o ajuste 2014 -> 2018.

---

## Criterios de aceite da segunda-feira

A segunda-feira pode ser considerada concluida quando:

- `metricas.py` calcula todas as metricas sem depender de um modelo especifico;
- a otimizacao de time-decay gera CSV completo e arquivo com fator escolhido;
- o walk-forward roda os tres folds temporais;
- 2025 entra apenas como validacao final, nao como escolha de hiperparametro;
- o relatorio menciona o recorte real `2018-2025`;
- MAE e Kendall tau sao comparados com as metas da arquitetura;
- as limitacoes ficam registradas.

---

## Riscos e ajustes recomendados

### Risco 1: diferenca entre 2014+ e 2018+

A arquitetura usa 2014+ como ideal por causa da era hibrida, mas o dataset final usa 2018+
por disponibilidade/qualidade das features FastF1. Isso nao invalida o cronograma, mas
precisa aparecer na metodologia e nas limitacoes.

### Risco 2: `season_factor` em modelos de arvore

Para validar 2025, `season_factor = 2025` fica fora do range observado no treino
2018-2024. Modelos de arvore nao extrapolam bem variaveis temporais numericas. Manter
por enquanto porque a RFE selecionou o conjunto atual, mas observar no tuning e na SHAP.

### Risco 3: top-3 estrita baixa

A top-3 atual exige acerto exato do conjunto de podio. Ela e util, mas severa. Para a
comparacao com Polishchuk [1], considerar adicionar depois uma metrica menos estrita:

- `top3_overlap`: media de quantos pilotos do podio real aparecem no top-3 previsto;
- `podium_recall`: proporcao dos pilotos reais de podio recuperados.

### Risco 4: margem muito pequena entre decays

`0.85` venceu `0.75` por diferenca minima de MAE. No texto, evitar vender isso como uma
descoberta forte. Melhor frase: "o grid-search indicou leve vantagem empirica para 0.85,
mantendo 0.75 como referencia teorica do RAPM".

---

## Proxima etapa apos concluir a segunda-feira

Na terca-feira, seguir o cronograma original:

- rodar XGBoost baseline ja validado pela infraestrutura;
- implementar/rodar Random Forest sem tuning;
- comparar metricas preliminares;
- corrigir eventuais problemas antes do Optuna.

Na quarta e quinta:

- Optuna para XGBoost;
- Optuna para Random Forest.

Na sexta:

- Ridge Regression baseline;
- tabela comparativa dos tres modelos.

No fim de semana:

- RFE final;
- re-rodar modelos com features definitivas;
- preparar feature importance e arquivos para comparacao com 2026.
