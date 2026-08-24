# Etapas de tratamento de dados para o Random Forest

Sequência de tratamento pensada especificamente para maximizar o desempenho do
Random Forest (sklearn), da base consolidada até o dataset pronto para o
modelo consumir.

## 1. Consolidação da base (pré-requisito)

Join de `resultados` + `pitstops` (agregado) + `fastf1_laps` (agregado) +
`fastf1_qualifying` + `fastf1_weather` (agregado) + `circuitos` (enriquecido
com `circuitos_manual.csv`) + `pilotos`, por `(season, round, driver_id)`.

## 2. Tratamento de DNF / classificação do target

Classificar `status` em DNF por falha mecânica vs erro/acidente vs terminou,
decidir se essas linhas entram como target degradado (posição = nº total de
pilotos, por ex.) ou são excluídas/tratadas à parte.

## 3. Valores ausentes — obrigatório

sklearn `RandomForestRegressor`/`Classifier` não aceita NaN. Isso é
inegociável, diferente de XGBoost/LightGBM. Pontos que fazem diferença real
na performance:

- Imputação hierárquica (mediana por circuito/temporada → temporada → global)
  em vez de média/mediana global simples — reduz viés introduzido pela
  imputação.
- Ajustar o imputador só com dados de treino (nunca com o dataset inteiro)
  para não vazar informação do futuro/validação.
- Considerar criar uma flag binária "era_ausente" antes de imputar certas
  colunas (ex: piloto estreante sem `recent_form`) — RF consegue aprender que
  "dado ausente" é, em si, um sinal (novato), coisa que se perde se só
  imputar silenciosamente.

## 4. Outliers — importante para RF, mais que para boosting

RF é sensível a outliers porque eles distorcem a média da folha e podem
forçar splits específicos para isolar um único ponto extremo, desperdiçando
profundidade da árvore. Trate assim:

- Detectar/remover erros de medição (ex: tempo de volta absurdo por bug de
  sensor) — sempre.
- Para outliers "legítimos" (ex: safety car, chuva extrema) prefira manter e
  sinalizar (flag categórica de contexto) em vez de remover, senão a RF perde
  capacidade de generalizar pra esses cenários.

## 5. Codificação categórica — aqui RF difere mais de XGBoost/LightGBM

- Variáveis com poucas categorias e sem ordem natural (ex: tipo de circuito
  0/1) → one-hot, sem problema.
- Alta cardinalidade (`driver_id` ~20-25, `constructor_id` ~10, `circuit_id`
  ~31): one-hot puro prejudica RF — cada split só enxerga uma dummy binária
  por vez, dilui a importância entre várias colunas esparsas e a árvore
  precisa de mais profundidade pra capturar o efeito. Prefira:
  - Target/mean encoding (com CV ou suavização, ajustado só no treino, pra
    não vazar) — ex: taxa histórica de pontos/vitórias do construtor.
  - Ou frequency encoding como alternativa mais simples e sem risco de
    leakage.
  - Isso normalmente já está coberto pelas features de RAPM/forma
    recente/taxa de DNF — ou seja, "encoding por desempenho histórico" em vez
    de one-hot bruto, que é o caminho certo para RF.
- Ordinal (composto de pneu: soft > medium > hard) → mantém como já é.

## 6. Escalonamento/normalização — pode pular para RF

Árvores dividem por limiar de valor, são invariantes a transformação
monotônica de escala. Z-score/MinMax não muda o desempenho da RF (só importa
se for comparar com modelo linear/distância). Não é etapa necessária pra esse
modelo especificamente.

## 7. Redundância / multicolinearidade

RF não quebra com colinearidade (diferente de regressão linear), mas dilui
importância entre features correlacionadas (ex: `driver_coef_rapm` e
`recent_form_5` correlacionados fazem a RF "dividir o crédito" entre as duas,
dificultando interpretação de importância). Vale checar matriz de correlação
e remover redundância óbvia antes da seleção final.

## 8. Seleção de features / remoção de ruído

RF tolera features irrelevantes, mas cada uma a mais é candidata a split em
cada árvore, aumentando variância e tempo de treino sem ganho. Usar
`feature_importances_` (impureza) ou, melhor, importância por permutação
(menos enviesada com alta cardinalidade) para podar as mais fracas antes de
fechar as features finais.

## 9. Balanceamento (se houver etapa de classificação, ex. prever DNF binário)

Se alguma sub-tarefa for classificação (não é o caso do `finish_position` em
si, que é regressão/ranking), RF se beneficia de `class_weight="balanced"` ou
reamostragem quando a classe minoritária (DNF) é rara.

## 10. Split temporal e validação

Não é "limpeza", mas afeta diretamente a performance percebida. Garantir que
toda feature histórica (RAPM, forma recente, etc.) é calculada só com dados
anteriores à corrida sendo prevista, e validar com walk-forward por
temporada/rodada — não faz diferença pro treinamento da árvore em si, mas
evita uma "melhora de performance" que na verdade é vazamento.

## 11. Validação final antes do modelo

Confirmar: zero NaN, sem colunas de variância quase-zero, sem colunas
proibidas (pós-corrida), ranges plausíveis — o mesmo tipo de checagem que o
`10_eda_validacao_dataset_tratado.py` já fazia.

## Resumo — o que muda de prioridade especificamente pra RF

1. Imputação é obrigatória, não opcional.
2. Outliers merecem mais cuidado do que dariam pra XGBoost/LightGBM.
3. Evitar one-hot em variáveis de alta cardinalidade — preferir encoding
   baseado em desempenho histórico.
4. Escalonamento pode ser descartado sem custo de performance.
