# 01 — Coleta de Dados

## Contexto

O primeiro problema a resolver foi construir uma base histórica de corridas de Fórmula 1 que fosse suficientemente rica para alimentar um modelo preditivo de posição final. Nenhuma API isolada oferece todos os dados necessários: resultados oficiais, tempos de volta, compostos de pneu, dados climáticos e posições de qualifying estão distribuídos em fontes distintas com schemas heterogêneos.

A hipótese subjacente é que o desempenho passado de pilotos e construtores, combinado com características do circuito e da corrida, contém sinal preditivo suficiente para estimar a posição final antes da largada.

---

## Fundamentação bibliográfica

| Decisão | Referência |
|---|---|
| Corte em 2014 (era híbrida) como critério de homogeneidade regulatória | Thomas et al. [7] — DNN Lap Time; Henderson et al. [9] — RAPM |
| Uso de Ergast como fonte de resultados históricos | Barra et al. [3], Ruan et al. [2], Henderson et al. [9] |
| FastF1 para telemetria de voltas e qualifying | Thomas et al. [7], Ruan et al. [2] |
| OpenF1 para dados em tempo real (2025-2026) | Adaptação própria — sem benchmark direto na literatura revisada |

A arquitetura propõe o corte em 2014 com a justificativa de que "a era híbrida garante homogeneidade regulatória" (seção 1, Fontes de Dados). A implementação adotou **2018** como corte efetivo — divergência documentada na seção de Avaliação Crítica abaixo.

---

## Implementação

### Scripts envolvidos

| Script | O que extrai | Saída |
|---|---|---|
| `src/extract_ergast_results.py` | Resultados por piloto por corrida (grid, posição, status, pontos) | `data/raw/ergast_2018_2024.csv` |
| `src/extract_ergast_2025.py` | Extensão dos resultados para 2025 | `data/raw/ergast_2025_results.csv` |
| `src/extract_ergast_pitstop.py` | Dados de pit stop (duração, volta, número de paradas) | `data/raw/ergast_pitstop_2018_2025.csv` |
| `src/extract_fastf1.py` | Qualifying (posição, tempos Q1/Q2/Q3), voltas de corrida (pneu, tempo, setor), clima | `data/raw/fastf1_qualifying_*.csv`, `fastf1_laps_*.csv`, `fastf1_weather_*.csv` |
| `src/extract_jolpica_circuits.py` | Metadados de circuito (altitude, tipo, comprimento, curvas) | `data/raw/jolpica_circuits.csv` |
| `src/extract_jolpica_drivers.py` | Metadados de piloto (nome, nacionalidade) | `data/raw/jolpica_drivers.csv` |
| `src/extract_openf1_race_data.py` | Resultados, stints, race control, clima — 2025 e 2026 | `data/raw/openf1_*.csv` |
| `src/extract_openf1_starting_grid_2025.py` | Grid de largada via endpoint `/starting_grid` da OpenF1 | `data/raw/openf1_starting_grid_2025.csv` |

### Por que três fontes?

Cada fonte cobre uma lacuna que as outras não preenchem:

| Fonte | O que oferece exclusivamente |
|---|---|
| **Ergast / Jolpica** | Resultado oficial histórico (posição final, status, pontos, grid). Série temporal completa 2018-2024. API estável com histórico consolidado. |
| **FastF1** | Telemetria detalhada por volta: composto de pneu, tempo por setor, pit in/out, status de pista (Safety Car via `TrackStatus`), dados de qualifying por piloto. Sem FastF1 não há `qualifying_position` nem `tire_compound_start`. |
| **OpenF1** | Dados ao vivo para 2025 e 2026. Ergast/Jolpica não tem cobertura confiável de temporadas em andamento. OpenF1 é a única fonte que permite validação walk-forward em 2025 e a futura análise de drift em 2026. |

### Por que 2018 e não 2014?

A arquitetura (seção 1) cita o corte em 2014 por homogeneidade regulatória, seguindo Thomas et al. [7] e Henderson et al. [9]. A implementação adotou **2018** pelo seguinte motivo prático: o FastF1 tem cobertura de qualifying e telemetria de voltas de forma confiável apenas a partir de 2018. Para temporadas anteriores, dados de qualifying estão incompletos ou ausentes na biblioteca.

O filtro está explícito no `src/limpeza_ergast_fastf1.py`, linha 229:

```python
ergast = ergast[ergast["season"] >= 2018].copy()
fastf1_laps = fastf1_laps[fastf1_laps["season"] >= 2018].copy()
```

Impacto da diferença: os benchmarks de Henderson et al. [9] e Thomas et al. [7] usam dados de 2014+. Qualquer comparação direta de MAE entre este projeto e esses papers precisa considerar que o conjunto de treino aqui é 4 anos menor.

(Isa)
### Teste empírico do impacto da janela histórica

Para avaliar se a ausência de temporadas anteriores a 2018 poderia explicar uma redução relevante nas métricas do modelo, foi criado um experimento controlado de janela histórica. O objetivo foi manter fixos o ano de teste, o modelo, as features finais e os hiperparâmetros, alterando apenas o primeiro ano disponível para treino. Como a base final do projeto não contém telemetria FastF1 completa para 2014-2017, o experimento não testa diretamente `2014-2024` contra `2018-2024`; ele testa uma pergunta metodologicamente equivalente dentro dos dados disponíveis: **o desempenho cai quando a janela histórica 2018-2024 é progressivamente encurtada?**

O experimento foi implementado em `src/experimento_janela_treino_2025.py` e documentado em `reports/modelagem/experimento_janela_treino_2025_relatorio.md`. Foi usado o LightGBM tunado, escolhido como modelo finalista principal na etapa de modelagem, com os mesmos hiperparâmetros salvos em `reports/modelagem/optuna_lightgbm_best_params.json` e `time-decay = 0.95`. O ano de teste foi mantido fixo em 2025.

| Experimento | Treino | Teste | Linhas de treino | MAE | R² | Top-3 accuracy |
|---|---|---:|---:|---:|---:|---:|
| A | 2018-2024 | 2025 | 2.524 | 2,3579 | 0,6477 | 0,2917 |
| B | 2019-2024 | 2025 | 2.189 | **2,3533** | **0,6505** | 0,2917 |
| C | 2020-2024 | 2025 | 1.829 | 2,3686 | 0,6451 | 0,2917 |
| D | 2021-2024 | 2025 | 1.546 | 2,3815 | 0,6429 | 0,2500 |

Os resultados não mostram uma queda monotônica ao remover apenas 2018. Na verdade, o experimento B (`2019-2024`) apresentou MAE e R² ligeiramente melhores que o experimento A (`2018-2024`), com diferença muito pequena: `-0,0046` de MAE e `+0,0028` de R². Isso indica que a inclusão de 2018, isoladamente, não foi determinante para o desempenho em 2025. Por outro lado, quando a janela fica mais curta (`2020-2024` e principalmente `2021-2024`), há piora gradual de MAE e R², além de queda do Top-3 accuracy no treino mais curto.

Assim, a interpretação metodológica é a seguinte: **não há evidência empírica de que a ausência de 2014-2017 seja a principal causa de eventuais métricas menores em relação aos benchmarks da literatura**. A comparação direta com trabalhos que treinam desde 2014 continua limitada, mas o experimento sugere que a qualidade e a consistência das features a partir de 2018 são mais importantes do que simplesmente adicionar anos antigos. Esse ponto é reforçado pela documentação oficial do FastF1: dados de timing, sessão, telemetria e posição estão disponíveis a partir de 2018; para temporadas anteriores, a biblioteca recorre a Ergast/Jolpica, com cobertura mais limitada e sem o mesmo nível de telemetria.

Essa conclusão também é coerente com a literatura de séries temporais e previsão esportiva. Em séries com mudança de regime, mais dados históricos nem sempre melhoram a previsão se os dados antigos representam relações diferentes das atuais. A Fórmula 1 passou por uma mudança técnica relevante em 2022, com novo regulamento aerodinâmico, pneus de 18 polegadas e combustível E10, o que reduz a comparabilidade direta com temporadas muito antigas. Além disso, estudos recentes sobre F1 reforçam que `qualifying_position` é um dos maiores determinantes da posição final; portanto, preservar features pré-corrida consistentes e completas pode ser mais relevante do que ampliar o histórico com temporadas de menor granularidade.

Fontes usadas para esta conclusão:

- Experimento local: `src/experimento_janela_treino_2025.py`.
- Métricas geradas: `reports/modelagem/experimento_janela_treino_2025_metricas.csv`.
- Relatório gerado: `reports/modelagem/experimento_janela_treino_2025_relatorio.md`.
- FastF1 Data Reference: timing data, session information, car telemetry e position data disponíveis a partir de 2018: https://docs.fastf1.dev/data_reference/index.html
- FastF1 API: para temporadas anteriores a 2018, o backend usado é Ergast, com limitações de dados locais/telemetria: https://docs.fastf1.dev/fastf1.html
- Scikit-learn `TimeSeriesSplit`: validação temporal deve treinar no passado e avaliar no futuro, evitando mistura temporal: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
- Liu et al. (2023), *Handling Concept Drift in Global Time Series Forecasting*: séries temporais podem sofrer mudança de distribuição ao longo do tempo, reduzindo acurácia quando o modelo assume estacionariedade: https://arxiv.org/abs/2304.01512
- Pesaran e Timmermann (2004), *How costly is it to ignore breaks when forecasting the direction of a time series?*: em presença de quebras estruturais, usar todo o histórico nem sempre é superior a usar uma janela recente selecionada: https://www.sciencedirect.com/science/article/abs/pii/S0169207003000682
- Formula 1 (2022), mudanças técnicas de 2022: novo conceito aerodinâmico, pneus de 18 polegadas e combustível E10: https://www.formula1.com/en/latest/article/2021-vs-2022-whats-changed-on-formula-1s-all-new-cars-for-this-season.1SrJj7fQgv4Iw2IxCTnb1l
- Weissbock (2025), *Evaluating the Predictive Power of Qualifying Performance in Formula One Grand Prix*: qualifying performance aparece como determinante forte da posição final: https://arxiv.org/abs/2507.10966

(/Isa)
### Construção do RaceID

A chave primária `RaceID` identifica de forma única cada participação de um piloto em uma corrida:

```python
ergast["RaceID"] = (
    ergast["driver_id"].astype(str)
    + "_"
    + ergast["season"].astype(int).astype(str)
    + "_"
    + ergast["round"].astype(int).astype(str)
)
```

**Exemplo:** `hamilton_2023_5` identifica Lewis Hamilton no GP 5 de 2023.

Unicidade garantida por: um piloto só pode ter uma posição final por corrida. A construção `driver_id + season + round` é irredutível — não existe combinação legítima duplicada. Após a criação, o script verifica e remove duplicatas: 0 duplicatas encontradas no dataset completo (confirmado no `relatorio_01`).

### Sincronização Ergast + FastF1

A integração é feita em `src/limpeza_ergast_fastf1.py`. O join usa `driver_id + season + round` como chave composta. O FastF1 usa códigos de três letras (ex: `HAM`) que precisam ser mapeados para os IDs do Ergast (ex: `hamilton`). (Isa) O mapeamento está hardcoded no dicionário `DRIVER_CODE_TO_ID` (43 entradas), cobrindo todos os códigos de pilotos presentes na base FastF1 utilizada no projeto.(/Isa)

O FastF1 é agregado por `RaceID` antes do join — cada piloto tem dezenas de voltas no FastF1, que são reduzidas a métricas por corrida (média de tempo de volta, composto predominante, contagem de pit stops, etc.).

Resultado do merge (do `relatorio_01`):
- 3.458 linhas no Ergast após concatenação 2018-2025
- 3.452 linhas no FastF1 após agregação por RaceID
- Os 6 registros sem correspondência FastF1 foram mantidos apenas na base integrada inicial, com valores ausentes nas colunas provenientes da FastF1. Como foram removidos antes da geração do dataset final de modelagem, eles não introduzem `NaN` no treinamento dos modelos. A limitação restante é documental e de rastreabilidade, pois esses casos precisam ser explicitamente reportados como exceções da integração entre fontes.

(Isa)
#### Registros sem correspondência FastF1

Durante a integração entre as bases Ergast/Jolpica e FastF1, foram identificados 6 registros presentes na base Ergast/Jolpica sem correspondência nas informações agregadas da FastF1. Esses registros foram preservados na base integrada inicial `historico_ergast_fastf1_limpo_2018_2025.csv`, mantendo valores ausentes nas colunas provenientes da FastF1.

A manutenção desses registros nessa etapa foi adotada por rastreabilidade, permitindo documentar que os pilotos constavam na fonte Ergast/Jolpica, mas não possuíam dados equivalentes nas features extraídas da FastF1. A análise individual da coluna `status` mostrou que esses registros estavam associados a situações excepcionais de corrida, como `Illness`, `Power Unit`, `Withdrew` e `Did not start`, indicando ausência de participação competitiva normal ou registro incompleto de desempenho em pista.

Posteriormente, esses 6 registros foram removidos nas etapas de tratamento de DNF/DNS/ausência de participação competitiva normal. A verificação confirmou que nenhum desses registros está presente nas bases posteriores `historico_dnf_excluded_2018_2025.csv`, `dataset_pre_features_2018_2025.csv`, `dataset_features_final_2018_2025.csv` ou `dataset_modelagem_2018_2025.csv`.

| Temporada | Corrida | Piloto | Status | Decisão |
|---:|---|---|---|---|
| 2021 | Abu Dhabi Grand Prix | mazepin | Illness | Removido antes da modelagem |
| 2022 | Saudi Arabian Grand Prix | tsunoda | Power Unit | Removido antes da modelagem |
| 2022 | Saudi Arabian Grand Prix | mick_schumacher | Withdrew | Removido antes da modelagem |
| 2023 | Singapore Grand Prix | stroll | Withdrew | Removido antes da modelagem |
| 2023 | Qatar Grand Prix | sainz | Did not start | Removido antes da modelagem |
| 2024 | São Paulo Grand Prix | albon | Did not start | Removido antes da modelagem |

Dessa forma, os registros sem correspondência FastF1 foram preservados inicialmente para controle e auditoria do processo de integração, mas não compõem o dataset final de modelagem. Essa decisão evita a introdução de valores ausentes nas features FastF1 e reduz o risco de que eventos não representativos de desempenho competitivo em pista distorçam o treinamento dos modelos.

### Síntese do tratamento dos registros sem FastF1

Como o join foi realizado com `how="left"`, os 6 registros foram preservados apenas na base integrada inicial para rastreabilidade. Entretanto, a verificação posterior confirmou que eles não seguem para o dataset final de modelagem. Assim, os valores ausentes nas colunas FastF1 ficam restritos à base inicial e não impactam o treinamento dos modelos.

(/Isa)

### Schema do OpenF1 vs. Ergast/FastF1

| Dimensão | Ergast/FastF1 | OpenF1 |
|---|---|---|
| Identificador de piloto | `driver_id` (string, ex: `hamilton`) | `driver_number` (inteiro, ex: 44) |
| Identificador de corrida | `season + round` | `meeting_key` (inteiro único) |
| Cobertura histórica | 2018–2024 (Ergast) | 2023+ (cobertura crescente) |
| Latência | Dados finais consolidados | Dados disponíveis durante/após a sessão |

(Isa)

A diferença de identificadores exige um mapeamento `driver_number → driver_id` para qualquer join com o pipeline histórico. A relação entre os schemas foi documentada em `docs/mapeamento_openf1_ergast.md`, gerado pelo script `src/mapear_openf1_ergast.py`. Já a aplicação operacional do mapeamento ocorre no pipeline OpenF1 por meio do dicionário `DRIVER_NUMBER_TO_ID_2025`.

Foi realizada uma verificação automática do mapeamento `driver_number → driver_id`, necessário para integrar dados da OpenF1 ao pipeline histórico baseado em Ergast/Jolpica. A checagem comparou os `driver_number` únicos presentes nos arquivos OpenF1 de 2025 com as chaves do dicionário `DRIVER_NUMBER_TO_ID_2025`, e comparou os `driver_id` resultantes com os identificadores presentes em `ergast_2025_results.csv`.

O resultado confirmou que o dicionário possui 21 entradas, cobrindo todos os 21 `driver_number` presentes nos arquivos OpenF1 de 2025. Não foram identificados números de pilotos sem mapeamento, `driver_id` mapeados sem correspondência na Ergast/Jolpica, nem `driver_id` da Ergast/Jolpica ausentes no dicionário.

Dessa forma, conclui-se que o mapeamento OpenF1 → Ergast/Jolpica está correto para a temporada de 2025. Para 2026, o dicionário ainda exige revisão manual, pois existem entradas provisórias associadas a novos números de pilotos, que não devem ser usadas como mapeamento definitivo.

(/Isa)

---

## Resultados obtidos

Do `relatorio_01_limpeza_ergast_fastf1_2018_2025.txt`:

| Métrica | Valor |
|---|---|
| Linhas brutas Ergast (2018-2025) | 3.458 |
| Linhas FastF1 brutas | 206.202 |
| Linhas FastF1 agregadas por RaceID | 3.452 |
| Registros removidos (nulos essenciais) | 0 |
| Duplicatas removidas | 0 |
| Registros com `grid_position = 0` (pit lane/DNS) | 45 |
| Registros sem correspondência FastF1 | 6 |
| **Linhas na base limpa 2018-2025** | **3.458** |

Distribuição por temporada:

| Temporada | Corridas-piloto |
|---|---|
| 2018 | 420 |
| 2019 | 420 |
| 2020 | 340 |
| 2021 | 440 |
| 2022 | 440 |
| 2023 | 440 |
| 2024 | 479 |
| 2025 | 479 |

2020 tem 340 registros (e não 440) por conta do calendário reduzido pela pandemia de COVID-19 — 17 corridas em vez de 20-22. Comprovação: https://ge.globo.com/motor/formula-1/noticia/mclaren-cre-em-uma-ou-duas-corridas-da-f1-canceladas-por-pandemia.ghtml

---

## Avaliação crítica

**Pontos fortes:**
- Três fontes complementares garantem cobertura completa das variáveis necessárias.
- Sem nulos nas colunas essenciais após a limpeza — base de partida limpa.
- RaceID com lógica simples, verificável e sem colisões.
- Mapeamento FastF1 → Ergast hardcoded é auditável e completo para o período coberto.

**Limitações:**
- Corte em 2018 (não 2014) reduz o período de treino em relação aos benchmarks da literatura. Comparações diretas de MAE com Henderson et al. [9] devem mencionar essa diferença, mas o experimento de janela histórica não encontrou evidência de que a ausência de 2014-2017 seja a principal causa de variação nas métricas.
- Os 6 registros sem correspondência FastF1 foram mantidos apenas na base integrada inicial, com valores ausentes nas colunas provenientes da FastF1. Como foram removidos antes da geração do dataset final de modelagem, eles não introduzem `NaN` no treinamento dos modelos. A limitação restante é documental e de rastreabilidade, pois esses casos precisam ser explicitamente reportados como exceções da integração entre fontes.
- O mapeamento `driver_number → driver_id` é hardcoded e precisa ser atualizado manualmente quando novos pilotos entram no grid (evidenciado pela necessidade de `DRIVER_NUMBER_TO_ID_2026_NEW` no `update_openf1_2026.py`).
- A Ergast API original foi descontinuada e migrada para Jolpica. Scripts que referenciam diretamente `ergast.com` precisariam ser atualizados.

(Isa)

**Riscos e validações adicionais:**

- Foram identificados 45 registros com `grid_position = 0` na base integrada inicial. Esses casos foram mantidos com `grid_position_zero_flag`, pois representam situações especiais de largada ou ausência de posição formal de grid. Como `grid_position = 0` não representa uma posição real de largada, o valor não deve ser usado diretamente no modelo, pois poderia ser interpretado como melhor que a pole position.

Foi realizado um experimento adicional para avaliar o tratamento dos registros com `grid_position = 0`. Foram comparados três cenários: o pipeline atual, a remoção dos registros marcados com `grid_position_zero_flag = 1` e a inclusão dessa flag como feature do modelo.

Os resultados indicaram que remover os registros com `grid_position_zero_flag = 1` gerou pequena melhora em MAE, RMSE e R², mas reduziu a acurácia Top-3 de 0,7361 para 0,6944. Como a previsão de pódio é um objetivo relevante do projeto, essa queda torna a remoção automática pouco recomendada.

A inclusão da variável `grid_position_zero_flag` apresentou o melhor equilíbrio geral, com melhora em MAE, RMSE e R² em relação ao cenário atual, embora com leve redução da acurácia Top-3. Dessa forma, conclui-se que os registros com `grid_position = 0` não devem ser removidos automaticamente. A abordagem mais adequada é mantê-los, tratar o valor zero antes da modelagem e, quando validado empiricamente, considerar a flag como variável explicativa auxiliar.

| Cenário | Linhas treino | MAE | RMSE | R² | Top-3 accuracy | Interpretação |
|---|---:|---:|---:|---:|---:|---|
| Atual, sem `grid_position` e sem flag | 2.524 | 2,3065 | 3,0460 | 0,6533 | **0,7361** | Melhor desempenho em Top-3 |
| Remover registros com `grid_position_zero_flag = 1` | 2.489 | **2,2995** | 3,0434 | 0,6539 | 0,6944 | Melhora erro médio, mas piora pódio |
| Incluir `grid_position_zero_flag` | 2.524 | 2,3028 | **3,0367** | **0,6554** | 0,7222 | Melhor equilíbrio geral |

(/Isa)

---

## Convergência com a literatura

| Aspecto | Alinhado | Divergente | Observação |
|---|---|---|---|
| Usar Ergast como fonte de resultados históricos | ✅ | — | Padrão da literatura [2][3][9] |
| FastF1 para qualifying e telemetria | ✅ | — | Seguido por [2][7] |
| Corte em 2014 (era híbrida) | — | ⚠️ | Implementado como 2018 por limitação do FastF1 |
| Chave piloto-corrida única | ✅ | — | Equivalente ao `RaceID` implícito em [9] |
| Join Ergast + FastF1 por piloto + round | ✅ | — | Mesma estratégia de [2] |
