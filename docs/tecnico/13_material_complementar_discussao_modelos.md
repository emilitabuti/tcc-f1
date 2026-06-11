# 13 — Material Complementar: Discussao Sobre Modelos, Linearidade e Literatura

## Objetivo

Este material complementar consolida a discussao metodologica sobre por que o
Ridge Regression superou os modelos de arvore no experimento oficial, por que
isso nao implica que a Formula 1 seja um fenomeno linear, quais modelos aparecem
nos trabalhos relacionados com dados pre-corrida e como mudancas regulatorias
podem ser tratadas como mudanca de dominio temporal.

O ponto central e que a comparacao deste TCC deve ser apresentada como uma
avaliacao empirica de familias de modelos usadas na literatura, sob a mesma
formulacao experimental:

- mesmo target oficial: `finish_position`;
- mesmas 13 features causais pre-corrida;
- mesma validacao temporal walk-forward;
- mesmas metricas oficiais: MAE, RMSE, R2 e Kendall tau;
- mesmo criterio composto, sem metrica top-3.

Assim, a conclusao nao depende da premissa de que LightGBM ou XGBoost deveriam
ser sempre superiores. A contribuicao do trabalho esta em testar modelos
representativos da literatura sob um protocolo temporal controlado.

---

## A predicao de F1 e linear?

Nao e correto afirmar que a predicao de Formula 1 seja linear em sentido geral.
A corrida real envolve dinamicas nao lineares, interacoes entre carros,
estrategia, clima, safety car, degradacao de pneus, acidentes, falhas mecanicas
e decisoes de box.

O que os resultados deste projeto sugerem e mais especifico:

> Na formulacao causal pre-corrida adotada neste TCC, grande parte do sinal
> preditivo disponivel e capturada por relacoes estruturais, aditivas e quase
> monotonicamente associadas ao resultado final.

Isso explica por que o Ridge Regression pode ir bem. O modelo nao esta
"descobrindo" toda a complexidade da corrida; ele esta explorando fatores que
ja carregam muito sinal antes da largada:

- `qualifying_position`;
- `constructor_coef_rapm`;
- `recent_form_5`;
- `driver_constructor_synergy`;
- `driver_coef_rapm`.

Essas variaveis resumem qualidade do carro, desempenho recente, posicao de
largada e efeitos historicos de piloto/construtor. Quando esse conjunto ja
explica a maior parte previsivel do resultado, modelos mais complexos podem ter
pouco ganho residual a explorar.

---

## Eventos de corrida e informacao nao observavel antes da largada

Parte importante do erro restante vem de eventos que acontecem durante a
corrida ou dependem de informacoes que nao estao plenamente disponiveis antes
da largada:

- entrada de safety car;
- acidentes e contatos;
- falhas mecanicas;
- chuva em momento especifico da prova;
- erros ou acertos de estrategia;
- undercut/overcut;
- pit stop lento;
- trafego depois da parada;
- degradacao real dos pneus em condicoes especificas.

Algumas features historicas tentam aproximar riscos medios, como
`constructor_dnf_rate`, mas elas nao informam se uma falha mecanica ou acidente
vai ocorrer naquela corrida especifica. Portanto, algoritmos como LightGBM e
XGBoost nao conseguem recuperar uma causalidade que nao esta representada nas
features.

Essa distincao e importante para a defesa:

> O modelo pre-corrida captura a estrutura previsivel da F1. O erro residual
> inclui eventos de corrida relevantes, mas nao conhecidos no momento da
> predicao.

---

## Por que Ridge pode superar LightGBM e XGBoost?

O resultado oficial mostra:

| Modelo | MAE | RMSE | R2 | Kendall tau | Score |
|---|---:|---:|---:|---:|---:|
| Ridge baseline | 2.2723 | 2.9574 | 0.6710 | 0.6543 | 0.5314 |
| LightGBM | 2.3172 | 3.0121 | 0.6587 | 0.6536 | 0.5279 |
| Random Forest | 2.3263 | 3.0121 | 0.6589 | 0.6503 | 0.5272 |
| XGBoost | 2.3415 | 3.0161 | 0.6578 | 0.6525 | 0.5269 |

A leitura recomendada e:

- Ridge foi o melhor modelo global no setup oficial.
- LightGBM foi o melhor modelo de arvore.
- Random Forest ficou praticamente empatado com LightGBM e superou levemente o
  XGBoost no score composto.
- XGBoost ficou marginalmente atras, mas sem fracasso metodologico.

Possiveis explicacoes:

1. **Sinal forte e quase linear.** `qualifying_position` e
   `constructor_coef_rapm` concentram grande parte da informacao util.
2. **Regularizacao ajuda em dados correlacionados.** Ridge usa penalizacao L2,
   reduzindo instabilidade entre features relacionadas.
3. **Dados tabulares temporais pequenos favorecem parcimonia.** O dataset tem
   poucas temporadas e validacao temporal estrita; modelos muito flexiveis podem
   aprender padroes especificos de uma era.
4. **Nao linearidade residual pode ser ruido.** Safety car, acidentes e falhas
   podem alterar fortemente o resultado, mas nao sao totalmente observaveis nas
   features pre-corrida.
5. **Random Forest e robusto por bagging.** A media de muitas arvores pode ser
   mais estavel do que boosting em cenarios pequenos e ruidosos.

Portanto, Ridge vencer nao significa que F1 seja simples. Significa que, sob
as restricoes causais do projeto, a parte previsivel antes da corrida esta muito
associada a efeitos estruturais capturados por um modelo linear regularizado.

---

## O que dizem os trabalhos relacionados?

Os trabalhos relacionados nao apontam para um unico algoritmo dominante. Eles
usam diferentes formulacoes do problema: regressao de posicao final,
classificacao de vencedor, classificacao de top-3, classes de resultado,
modelagem ordinal, decomposicao piloto/construtor e modelos tabulares.

| Referencia | Tipo de problema | Modelos usados |
|---|---|---|
| Rane (2025) | Decomposicao piloto/construtor e predicao de resultado | Ridge / regressao linear regularizada com time decay |
| Van Kesteren & Bergkamp (2022/2023) | Modelagem de ranking/posicoes finais | Bayesian multilevel rank-ordered logit |
| Weissbock & Mills (2025) | Poder preditivo do qualifying | Ordinal Logistic Regression e analise estatistica |
| Krzyszton & Smolka (2026) | Classificacao de resultado | SVM, Gradient Boosting e Random Forest com Optuna |
| Thomas et al. / Preprint TabNet (2025) | Posicao final e pontos de construtor | TabNet |
| Sobrie (2020), citado por Weissbock & Mills | Classificacao top-3 | Decision Trees, Random Forest, AdaBoost, Gradient Boosting e XGBoost |
| Nigro (2020) | Predicao de vencedor | Logistic/Linear Regression, Random Forest, SVM e redes neurais |
| Stoppels (2017) | Predicao de resultados finais | Redes neurais artificiais e regressao logistica multiclasses |

Essa diversidade fortalece a decisao de comparar familias de modelos, em vez
de assumir que LightGBM ou XGBoost deveriam ser vencedores por padrao.

---

## Dados pre-corrida e comparabilidade

Muitos estudos de F1 usam variaveis que podem nao estar disponiveis antes da
corrida, como telemetria, voltas completadas, overtakes, informacoes de treino
ou dados intra-corrida. Este TCC adota uma formulacao mais restritiva:

- prever `finish_position` antes da corrida;
- usar somente features disponiveis antes da largada ou calculadas com historico
  anterior;
- validar por walk-forward temporal;
- remover top-3 como metrica oficial, porque top-3 e usualmente tratado na
  literatura como classificacao direta.

Por isso, comparacoes com trabalhos de classificacao de podio ou vencedor devem
ser apresentadas como referencias de contexto, nao como metas diretas para a
regressao de posicao final.

---

## Transfer learning, mudancas regulatorias e domain shift

Mudancas regulatorias em F1 podem ser entendidas como um problema de mudanca de
dominio temporal. O passado continua informativo, mas a relacao entre carro,
piloto, classificacao, ritmo de corrida e confiabilidade pode mudar apos uma
nova era tecnica.

Exemplos de mudancas relevantes:

- 2014: inicio da era hibrida;
- 2017: mudancas aerodinamicas amplas;
- 2022: novo regulamento de efeito solo;
- 2026: nova unidade de potencia e novo pacote tecnico.

Na literatura especifica de F1, ha pouca evidencia direta de transfer learning
formal para mudancas regulatorias. A abordagem mais defensavel para este TCC e
tratar o problema como adaptacao temporal:

> adaptacao a mudancas regulatorias por walk-forward, time decay e regularizacao.

Abordagens possiveis:

| Abordagem | Ideia | Adequacao ao projeto |
|---|---|---|
| Time decay / recency weighting | Dar mais peso a corridas recentes | Alta; ja alinhada ao pipeline |
| Ridge/Bayesian dinamico | Atualizar efeitos de piloto/construtor com regularizacao | Alta; combina com RAPM |
| Janela temporal recente | Treinar apenas em eras proximas | Media/alta; reduz historico obsoleto |
| LightGBM/XGBoost com sample weights | Usar boosting com pesos temporais | Media/alta; ja compativel com o pipeline |
| TrAdaBoost / transfer boosting | Reponderar exemplos antigos conforme ajudam ou atrapalham | Promissora, mas mais complexa |
| Deep tabular transfer learning | Pre-treinar em dados antigos e fine-tunar na era nova | Possivel, mas exige mais dados e validacao |
| Domain-adversarial learning | Aprender representacoes menos dependentes da era | Avancada; baixa prioridade para este TCC |
| Online learning | Atualizar o modelo corrida a corrida | Conceitualmente forte para F1 |

Para dados tabulares pequenos, a estrategia mais robusta costuma ser comecar
por metodos simples e controlaveis: walk-forward, time decay, regularizacao e
pesos temporais. Modelos profundos de transfer learning podem ser promissores,
mas exigem mais cuidado para nao introduzir instabilidade ou overfitting.

---

## Narrativa recomendada para o TCC

Em vez de defender que LightGBM e XGBoost deveriam ser os melhores, a narrativa
mais forte e:

> Este trabalho compara familias de modelos usadas na literatura de predicao de
> resultados de Formula 1 sob uma mesma formulacao experimental causal e
> temporal. Embora modelos de boosting sejam frequentemente competitivos em
> dados tabulares, o Ridge Regression apresentou melhor desempenho global neste
> setup, sugerindo que grande parte do sinal preditivo pre-corrida esta
> concentrada em efeitos estruturais de largada, construtor, forma recente e
> coeficientes RAPM. Entre os modelos de arvore, LightGBM foi o melhor, seguido
> de Random Forest e XGBoost em empate tecnico.

Essa leitura transforma o resultado em contribuicao metodologica:

- nao ha escolha de algoritmo por popularidade;
- todos os modelos sao avaliados no mesmo protocolo;
- o resultado empirico decide a hierarquia;
- Ridge vira baseline forte e interpretavel;
- LightGBM permanece como melhor arvore;
- Random Forest demonstra robustez;
- XGBoost permanece como comparativo relevante da literatura.

---

## Texto pronto para inserir na discussao

Os resultados indicam que o melhor desempenho global foi obtido pelo Ridge
Regression, seguido por LightGBM, Random Forest e XGBoost. Esse resultado nao
deve ser interpretado como evidencia de que a Formula 1 seja um fenomeno
linear. A corrida em si envolve processos complexos e nao lineares, como
safety cars, estrategia de pneus, acidentes, falhas mecanicas e condicoes
climaticas. Entretanto, como este trabalho adota uma formulacao causal
pre-corrida, o modelo observa apenas informacoes disponiveis antes da largada.
Nesse contexto, grande parte do sinal previsivel esta associada a fatores
estruturais, como posicao de classificacao, forca do construtor, forma recente
e coeficientes historicos de piloto/construtor.

A literatura relacionada tambem nao estabelece um unico algoritmo dominante
para predicao de resultados de Formula 1. Trabalhos distintos usam modelos
lineares regularizados, modelos ordinais, modelos bayesianos de ranking,
Random Forest, Gradient Boosting, XGBoost, redes neurais e TabNet, muitas vezes
em formulacoes diferentes, como predicao de vencedor, classificacao de podio,
classes de resultado ou regressao de posicao final. Por isso, a principal
contribuicao experimental deste trabalho esta em comparar diferentes familias
de modelos sob o mesmo protocolo temporal e causal.

Sob essa perspectiva, o desempenho superior do Ridge sugere que as features
selecionadas capturam de forma eficiente os componentes mais estaveis do
resultado pre-corrida. Ja LightGBM, Random Forest e XGBoost continuam
metodologicamente relevantes por representarem familias amplamente usadas na
literatura e por testarem a existencia de sinal nao linear adicional. O fato de
LightGBM ser o melhor modelo de arvore reforca sua utilidade, mas os resultados
nao justificam uma narrativa centrada exclusivamente em boosting.

---

## Fontes consultadas

- Rane, S. "Predicting Formula 1 Race Outcomes: Decomposing the Roles of Drivers and Constructors through Linear Modeling". https://arxiv.org/abs/2508.00200
- Van Kesteren, E.-J.; Bergkamp, T. "Bayesian Analysis of Formula One Race Results: Disentangling Driver Skill and Constructor Advantage". https://arxiv.org/abs/2203.08489
- Weissbock, J.; Mills, S. "Evaluating the Predictive Power of Qualifying Performance in Formula One Grand Prix". https://arxiv.org/abs/2507.10966
- Krzyszton, S.; Smolka, J. "Application of machine learning for predicting Formula 1 race results". https://ph.pollub.pl/index.php/jcsi/article/view/8462
- Thomas et al. "The Use of Machine Learning in Predicting Formula 1 Race Outcomes". https://www.preprints.org/manuscript/202504.1471
- Stoppels, E. "Predicting Formula One Race Results". https://essay.utwente.nl/74765/1/FinalThesisEloyStoppelsNoCompany.pdf
- scikit-learn. "Ridge regression and classification". https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression-and-classification
- scikit-learn. "RandomForestRegressor". https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
- XGBoost documentation. "Notes on parameter tuning". https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html
- LightGBM documentation. "Parameters Tuning". https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
- Tabular transfer learning, ICLR 2023. https://openreview.net/forum?id=b0RuGUYo8pA
