# 12 — Baselines de Literatura para Predição de Resultado de F1

## Objetivo

Este documento consolida referências encontradas em busca web/acadêmica sobre previsão de resultados de Fórmula 1, com foco em estudos comparáveis ao pipeline deste projeto:

- previsão de posição final ou ranking de chegada;
- uso de features disponíveis antes da corrida, quando informado;
- métricas quantitativas que possam servir de baseline;
- distinção entre regressão de posição, classificação de pódio e classificação por faixas.

A conclusão principal é que poucos trabalhos fazem exatamente a mesma tarefa deste projeto: regressão walk-forward temporal de `finish_position` com controle explícito de leakage. Por isso, os baselines abaixo devem ser usados com graus diferentes de comparabilidade.

---

## Por que regressão causal?

O pipeline principal deste projeto usa regressão causal porque o objetivo definido na arquitetura é prever `finish_position` antes da corrida. A palavra "causal" não significa estimar efeito causal no sentido econométrico estrito; significa que todas as features usadas pelo modelo precisam estar disponíveis antes da corrida alvo ou ser calculadas somente com histórico anterior.

Essa decisão se apoia em quatro blocos da literatura:

- Thomas et al. / TabNet: trata o resultado como predição de posição final e reporta métricas de regressão.
- Henderson/Rane: usa regressão com time-decay e decomposição piloto/construtor para modelar resultado de corrida.
- Van Kesteren & Bergkamp: modela posições finais como ranking, não como classificação binária de pódio.
- Weissbock & Mills: mostra que qualifying é forte preditor de posição final, justificando features pré-corrida.

Assim, o desenho metodológico do projeto é:

- prever posição final ou ranking;
- respeitar a ordem temporal por walk-forward;
- excluir ou recalcular variáveis que só seriam conhecidas durante/depois da corrida;
- deixar classificação de pódio/top-3 fora do pipeline principal.

Esse desenho é mais restritivo que muitos trabalhos de pódio, mas é mais adequado para a pergunta real de previsão pré-corrida.

---

## Metas comparáveis recomendadas

As metas originais continuam úteis como referência aspiracional, mas a comparação principal deve considerar o tipo de problema. Após a revisão, as metas comparáveis recomendadas são:

| Métrica | Meta comparável | Papel na avaliação |
|---|---:|---|
| MAE | ≤ 2.35 | Métrica principal de erro médio em posições |
| RMSE | ≤ 3.0 | Penaliza erros grandes; meta ainda alinhada ao TabNet |
| R² | ≥ 0.65 ou ≥ 0.66 | Meta realista para setup causal com features pré-corrida |
| Kendall τ | ≥ 0.60 | Métrica principal de qualidade de ranking |

Top-3 foi removido das métricas oficiais. Trabalhos que reportam acurácia de pódio tratam outro problema, normalmente classificação direta, e não devem ser usados como critério de sucesso da regressão causal.

---

## Baselines Mais Comparáveis

| Referência | Tipo de problema | Dados / validação | Métricas reportadas | Comparabilidade com este projeto |
|---|---|---|---|---|
| Thomas et al. / Preprints.org, "The Use of Machine Learning in Predicting Formula 1 Race Outcomes" | Regressão de posição final com TabNet | 2010-2022 treino, 2023 teste cronológico | RMSE 2.87, MAE 2.17, R² 0.75, correlação 0.87 | Alta para métricas, mas com cautela: o texto menciona `laps completed` e `IsOvertake`, que podem não ser estritamente pré-corrida |
| Rane, "Predicting Formula 1 Race Outcomes: Decomposing the Roles of Drivers and Constructors through Linear Modeling" | Regressão linear/RAPM com time-decay | Era híbrida 2014-2024 | Construtor explica 64.0% da variância | Alta para fundamentar RAPM e força do construtor; baixa para comparar MAE/RMSE, pois o resumo não reporta essas métricas |
| van Kesteren & Bergkamp, "Bayesian Analysis of Formula One Race Results" | Rank-ordered logit para posições finais | Era híbrida 2014-2021; não-finalistas removidos | Construtor explica ~88% da variância dos resultados | Alta para justificar modelagem de ranking e remoção de DNFs; baixa como baseline numérico direto |
| Weissbock & Mills, "Evaluating the Predictive Power of Qualifying Performance..." | Regressão logística ordinal / análise de ranking | ~7.800 observações de fim de semana | Spearman por era turbo-híbrida ~0.777 entre qualifying e chegada | Alta para justificar `qualifying_position` como feature dominante; não reporta MAE/RMSE |
| Krzysztoń & Smołka, "Application of machine learning for predicting Formula 1 race results" | Classificação multi-classe: vencedor, top-3, pontos, sem pontos | OpenF1 2023-2024; 5-fold CV | F1-score 77.83% ± 4.18%; accuracy 82.25% ± 3.83%; RMSE de classe 0.4719 ± 0.0609 | Média/baixa: prevê classes de resultado, não posição final contínua |

---

## Referência Direta para Metas de Regressão

O baseline mais diretamente comparável em termos de métricas é o estudo com TabNet:

| Métrica | TabNet literatura | Nosso baseline oficial LightGBM | Nosso melhor modelo de regressão oficial |
|---|---:|---:|---:|
| MAE | 2.17 | 2.3172 | 2.2723 Ridge |
| RMSE | 2.87 | 3.0121 | 2.9574 Ridge |
| R² | 0.75 | 0.6587 | 0.6710 Ridge |
| Correlação | 0.87 | não usada como métrica final | não usada como métrica final |

Leitura:

- O nosso MAE está próximo do baseline TabNet, mas ainda acima.
- O RMSE do Ridge fica abaixo de 3.0; as árvores ficam ligeiramente acima.
- O maior gap continua sendo R²: ~0.66 neste projeto contra 0.75 no TabNet.
- A comparação deve ser apresentada com cautela porque o estudo TabNet menciona variáveis como voltas completadas e indicador de ultrapassagem, que podem refletir informação intra/pós-corrida se usadas como features.

---

## Referências de Ranking e Estrutura do Problema

### RAPM / decomposição piloto-construtor

Rane (2025) usa ridge regression com time-decay e LOESS para prever resultados na era híbrida. O resultado mais importante para este projeto é que o construtor explica 64.0% da variância em resultados de corrida.

Van Kesteren & Bergkamp (2022/2023) usam um modelo Bayesian multilevel rank-ordered logit para modelar posições finais individuais. Eles removem não-finalistas da análise, assim como este projeto remove DNFs/DSQs para manter a coerência do target de posição final. O estudo conclui que aproximadamente 88% da variância dos resultados é explicada pelo construtor.

Essas duas referências sustentam três decisões do pipeline:

- incluir `constructor_coef_rapm`;
- tratar piloto e construtor como componentes separados;
- remover DNFs/DSQs para modelar posição final condicional a completar a corrida.

### Qualifying como principal preditor

Weissbock & Mills mostram que a performance de qualifying é o determinante mais forte da posição final, superando prática e posição de largada. Na era turbo-híbrida, a associação reportada entre qualifying position e finish position é alta, com Spearman em torno de 0.777.

Isso apoia diretamente a inclusão e dominância de `qualifying_position` no nosso modelo. A concentração de importância dessa feature não é necessariamente um erro: ela é coerente com a literatura.

---

## Referências Pouco Comparáveis

### Classificação de pódio/top-3

Foi feita uma busca específica por trabalhos que combinassem simultaneamente:

- previsão pré-corrida ou temporalmente causal;
- regressão/ranking de posição final;
- métrica de top-3 derivada da predição de posição.

Não foi encontrado, entre as referências acadêmicas revisadas, um estudo diretamente equivalente ao pipeline deste projeto medindo top-3 exato a partir de regressão causal de `finish_position`. Por isso, top-3 foi retirado das métricas oficiais do pipeline principal.

O padrão encontrado na literatura é separar os problemas:

- trabalhos de regressão/ranking reportam MAE, RMSE, R², correlação ou decomposição de variância;
- trabalhos de pódio/top-3 tratam o problema como classificação binária ou multi-classe;
- benchmarks relacionais como RelBench possuem tarefas separadas para `driver-top3` (classificação, AUROC) e `results-position`/`driver-position` (regressão, MAE).

Portanto, não há uma meta acadêmica robusta para dizer que uma regressão causal de posição final deveria atingir 70% de top-3 exato. A comparação metodologicamente correta é:

- regressão causal → comparar MAE, RMSE, R² e métricas de ranking;
- pódio/top-3 → comparar apenas com modelos treinados especificamente para classificação de pódio.

### Classes de resultado

Krzysztoń & Smołka (2026) alcançam F1-score 77.83% e accuracy 82.25%, mas o problema é multi-classe:

- vencedor;
- top-3;
- pontos;
- sem pontos.

Isso não é equivalente a prever `finish_position` contínua. Portanto, serve como referência de que classificação por faixas pode ter desempenho alto, mas não deve ser usada como meta direta para MAE/RMSE/R²/Kendall τ.

### Polishchuk 78% top-3

O artigo de Polishchuk não é acadêmico e usa classificação direta de pódio. Além disso, a métrica reportada parece medir acerto de vagas individuais de pódio, não igualdade exata do conjunto inteiro de três pilotos. O texto também menciona dataset com features pré-corrida e pós-corrida, então não deve ser usado como baseline acadêmico direto.

Para este TCC, Polishchuk pode ser citado apenas como exemplo de formulação alternativa de classificação de pódio, não como comparação direta com o pipeline principal.

---

## Baseline Recomendado para a Defesa

Para comparar o pipeline atual com a literatura, usar a seguinte narrativa:

1. **Regressão de posição final:** comparar principalmente com o TabNet.
   - Literatura: MAE 2.17, RMSE 2.87, R² 0.75.
   - Nosso melhor modelo oficial: MAE 2.2723, RMSE 2.9574, R² 0.6710 no Ridge baseline.
   - Nosso melhor modelo de árvore oficial: MAE 2.3172, RMSE 3.0121, R² 0.6587 no LightGBM.

2. **Ranking/estrutura do problema:** usar Rane e van Kesteren & Bergkamp.
   - Construtor explica 64% a 88% da variância, sustentando o RAPM e a importância de features de construtor.

3. **Feature dominante:** usar Weissbock & Mills.
   - Qualifying é o principal preditor da posição final, justificando a dominância de `qualifying_position`.

4. **Top-3:** não usar como métrica oficial.
   - Trabalhos com 70%+ geralmente tratam pódio como classificação direta ou classes agregadas.
   - Não foi encontrado baseline acadêmico equivalente para top-3 exato derivado de regressão causal de posição final.

---

## Fontes Consultadas

- Thomas et al. / Preprints.org. "The Use of Machine Learning in Predicting Formula 1 Race Outcomes". https://www.preprints.org/manuscript/202504.1471
- Saurabh Rane. "Predicting Formula 1 Race Outcomes: Decomposing the Roles of Drivers and Constructors through Linear Modeling". https://arxiv.org/abs/2508.00200
- Erik-Jan van Kesteren; Tom Bergkamp. "Bayesian Analysis of Formula One Race Results: Disentangling Driver Skill and Constructor Advantage". https://arxiv.org/abs/2203.08489
- Joshua Weissbock; Shirley Mills. "Evaluating the Predictive Power of Qualifying Performance in Formula One Grand Prix". https://arxiv.org/abs/2507.10966
- Sylwia Krzysztoń; Jakub Smołka. "Application of machine learning for predicting Formula 1 race results". https://ph.pollub.pl/index.php/jcsi/article/view/8462
- Illia Polishchuk. "Predicting F1 Podiums with 78% Accuracy Using Machine Learning and Real Race Data". https://ipolishchuk22.medium.com/predicting-f1-podiums-with-78-accuracy-using-machine-learning-and-real-race-data-0de3bcd6d2c4
- RelBench `rel-f1` benchmark. https://relbench.stanford.edu/datasets/rel-f1
