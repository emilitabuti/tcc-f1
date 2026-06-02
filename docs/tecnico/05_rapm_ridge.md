# 05 — RAPM Ridge

## Contexto

O problema central desta etapa é representar pilotos e construtores como variáveis numéricas sem usar OHE (alta dimensionalidade) nem Label Encoding (ordem artificial). A solução é estimar um coeficiente de habilidade para cada entidade usando o histórico de resultados — um número que cresce quando o piloto performa melhor que o esperado e decresce quando performa abaixo.

Esse coeficiente é calculado separadamente para cada corrida, usando apenas o histórico anterior a ela, garantindo que não haja informação do futuro na feature.

---

## Fundamentação bibliográfica

| Decisão | Referência |
|---|---|
| RAPM como técnica de decomposição de desempenho | Henderson et al. [9] — "Predicting Formula 1 Race Outcomes: Decomposing the Roles of Drivers and Constructors through Linear Modeling" |
| Construtor explica ~88% da variância na era híbrida | Snoeks [10] — "Bayesian Analysis of Formula One Race Results: Disentangling Driver Skill and Constructor Advantage" |
| Time-decay fator 0.75 por temporada | Henderson et al. [9] — valor ótimo encontrado no RAPM paper |
| Ridge Regression para regularizar coeficientes | Henderson et al. [9]; princípio geral de regularização L2 |

Henderson et al. [9] é o paper central desta etapa — é a única referência da revisão que cita explicitamente a necessidade de recalibração para mudanças regulatórias, sendo o único que trata causalmente o problema de decomposição piloto/construtor no contexto temporal da F1.

---

## O que é RAPM

**RAPM** (*Regularized Adjusted Plus-Minus*) é uma técnica originada no basquete para estimar a contribuição individual de cada jogador ao resultado do time, descontando o efeito dos demais jogadores presentes em quadra simultaneamente.

No basquete, a equação é:
```
Δscore = Σ(coef_jogador_i × presença_i) + intercepto + ε
```

Cada jogador recebe um indicador binário: 1 se estava em quadra, -1 se estava no banco. O Ridge regulariza os coeficientes para evitar overfitting quando algum jogador tem poucas amostras.

---

## Adaptação para Fórmula 1 — como difere do RAPM original

Na F1, cada piloto corre sozinho pelo seu carro — não há sobreposição de equipes em pista no mesmo instante de forma modelável. A adaptação de Henderson et al. [9] usa uma formulação mais simples:

```
-finish_position = coef_piloto + coef_construtor + intercepto + ε
```

Cada corrida gera uma linha com indicadores binários:
- 1 na coluna do piloto que correu
- 1 na coluna do construtor daquele piloto
- 0 em todas as demais colunas

A matriz é esparsa (2 valores não-zero por linha em ~50+ colunas). O Ridge com penalização L2 regulariza coeficientes de entidades com poucas corridas para próximo de zero.

**O que difere do RAPM estrito:**

| Aspecto | RAPM original (basquete) | Implementação F1 |
|---|---|---|
| Simultaneidade | Ajusta para quem está em quadra ao mesmo tempo | Não há simultaneidade — piloto e construtor por corrida |
| Confusão com posição de largada | `grid_position` incluída como covariável em alguns estudos | **Não incluída** — confundidor não controlado |
| Regularização | Ridge ou LASSO | Ridge (L2) |
| Temporal | Não endereçado no basquete | Time-decay por temporada |

**Por que a ausência de `grid_position` como covariável é relevante para a defesa:**

Sem controlar `grid_position` no RAPM, os coeficientes de pilotos e construtores capturam conjuntamente o desempenho na corrida e o efeito de largar na frente. Um construtor que consistentemente coloca seus pilotos no P1 do grid (via carro superior no qualifying) terá `constructor_coef_rapm` alto — mas parte disso vem da vantagem de largada, não apenas do desempenho em corrida. A implementação é mais próxima de um **Fixed Effects Model** do que RAPM estrito. Isso está documentado como limitação — é uma simplificação consciente e defensável dado que o RAPM é usado como *feature de entrada* no modelo principal, não como modelo final.

---

## Implementação

**Script:** `src/rapm_ridge.py`

### Fluxo corrida a corrida (causalidade estrita)

```
Para cada corrida r em ordem cronológica:
    train = todas as corridas com race_order < r
    if len(train_races) < min_races_train (=1):
        coef_piloto = 0.0   # cold start
        coef_construtor = 0.0
    else:
        X = matriz binária esparsa de train
        y = -finish_position de train
        w = time_decay(train, r)
        model = Ridge(alpha=10.0).fit(X, y, sample_weight=w)
        coef_piloto[r] = model.coef_[índice do piloto]
        coef_construtor[r] = model.coef_[índice do construtor]
```

173 corridas no período 2018-2025 → 173 modelos Ridge treinados. Cada piloto recebe o coeficiente calculado *antes* de correr aquela corrida.

### Por que target = `-finish_position`?

`finish_position = 1` é o melhor resultado. Em regressão linear, coeficientes maiores implicam valores maiores do target. Se o target fosse `+finish_position`, um coeficiente alto significaria piloto ruim (posições altas = posições ruins). Invertendo o sinal, coeficiente maior → melhor desempenho histórico estimado.

Exemplo: Hamilton com `driver_coef_rapm = 2.3` significa que, historicamente, ele contribui com ~2.3 posições a mais que o esperado pelo construtor e intercepto.

### Time-decay

Corridas mais antigas recebem peso menor. A fórmula:

```python
distancia = season_atual - season_treino  # em temporadas
peso = decay ^ distancia                  # decay = 0.75
```

| Temporada | Distância (em 2025) | Peso |
|---|---|---|
| 2025 | 0 | 1.000 |
| 2024 | 1 | 0.750 |
| 2023 | 2 | 0.563 |
| 2022 | 3 | 0.422 |
| 2021 | 4 | 0.316 |
| 2018 | 7 | 0.133 |

O valor 0.75 vem diretamente de Henderson et al. [9], que o identificaram como o fator ótimo para o dataset de F1. A unidade é **por temporada** (`decay_unit = "season"`), não por corrida — isso é coerente com o fato de que regulamentos e equipes mudam entre temporadas, não entre corridas individuais dentro de uma temporada.

**Diferença importante:** o RAPM usa decay=0.75, mas o walk-forward de modelagem usa decay=0.95 (otimizado empiricamente). São dois contextos diferentes — o RAPM calcula coeficientes de habilidade a longo prazo; o walk-forward pondera corridas recentes para a previsão imediata. A diferença está documentada no documento 08 (Walk-Forward).

### Regularização Ridge — alpha = 10.0

O alpha controla a força da penalização L2. Alpha maior → coeficientes mais próximos de zero → mais conservadores.

```python
model = Ridge(
    alpha=10.0,
    fit_intercept=True,
    solver="auto",
    random_state=42,
)
```

**O alpha=10.0 não foi tunado.** A arquitetura previa grid-search do alpha via validação cruzada temporal, mas o valor default foi mantido. O manifesto registra `"alpha": 10.0` sem documentação de processo de seleção.

**Por que isso é defensável:** alpha conservador (alto) é preferível quando há entidades com poucas corridas — pilotos estreantes ou construtores em anos iniciais. Um alpha pequeno permitiria coeficientes extremos para entidades com 1-2 corridas, gerando instabilidade. Alpha=10 força conservadorismo, o que é metodologicamente justificável para features de entrada (o modelo principal irá ponderar essas features conforme a importância real).

**O que deveria ter sido feito:** grid-search em {0.1, 1, 5, 10, 50, 100} avaliando a correlação dos coeficientes com o target no conjunto de validação temporal. Isso permanece como limitação metodológica documentada.

### Cold-start

A primeira corrida de qualquer piloto ou construtor sem histórico recebe coeficiente `0.0`. O relatório registra `1` corrida com cold-start inicial (a primeira corrida de toda a base — a corrida 1 de 2018, sem nenhuma corrida anterior).

Para pilotos estreantes em corridas posteriores: o RAPM roda corrida a corrida, então um piloto que estreou na corrida 50 recebe cold-start=0.0 na corrida 50 e passa a ter histórico a partir da corrida 51. Esse comportamento é rastreado pela flag `rapm_cold_start_flag` nos arquivos de saída.

---

## Resultados obtidos

Do `relatorio_10_rapm_ridge.txt` e `manifest_rapm_ridge.json`:

| Métrica | Valor |
|---|---|
| Total de corridas processadas | 173 |
| Total de linhas de coeficientes (pilotos) | 2.943 |
| Total de linhas de coeficientes (construtores) | 2.943 |
| Corridas com cold-start | 1 |
| Alpha Ridge | 10.0 |
| Time-decay | 0.75 por temporada |
| LOESS aplicado | Não |
| Data de execução | 22/05/2026 |

---

## Merge dos coeficientes no dataset principal

Os coeficientes são integrados no Feature Engineering via merge por `RaceID`:

```python
# Contrato de merge (do manifest):
# drivers:      [season, round, RaceID, driver_id,      driver_coef_rapm]
# constructors: [season, round, RaceID, constructor_id, constructor_coef_rapm]
```

O merge usa `validate="many_to_one"` para garantir que não há duplicação de linhas. Se um `RaceID` não encontrar coeficiente no arquivo RAPM, recebe `0.0` via `fillna(0)` — comportamento de cold-start.

---

## Avaliação crítica

**Pontos fortes:**
- Causalidade estrita: cada coeficiente usa somente histórico anterior. Verificável no código: `train = df[df["race_order"] < current_order]`.
- 173 modelos Ridge independentes — os coeficientes evoluem ao longo do tempo, capturando melhora/piora de pilotos e equipes.
- Extensível: novos pilotos e construtores recebem cold-start=0.0 sem quebrar o pipeline.
- Manifesto JSON registra todos os parâmetros para reprodutibilidade.

**Limitações:**
- Alpha=10.0 não tunado — a escolha é defensável mas não otimizada.
- `grid_position` não incluída como covariável — os coeficientes capturam parte do efeito de largada, não apenas o desempenho em corrida.
- LOESS disponível mas não usado — uma suavização dos coeficientes ao longo do tempo poderia reduzir oscilações pontuais, mas tornaria a metodologia mais complexa.
- Correlação esperada com o target: Henderson et al. [9] reportam que o construtor explica ~64% e o piloto ~36% da variância. No dataset deste projeto: `constructor_coef_rapm` tem r=-0.683 com `finish_position` e `driver_coef_rapm` tem r=-0.599 — valores consistentes com a literatura.

---

## Convergência com a literatura

| Aspecto | Alinhado | Divergente | Observação |
|---|---|---|---|
| Ridge Regression para coeficientes | ✅ | — | Henderson et al. [9] |
| Time-decay = 0.75 por temporada | ✅ | — | Valor do RAPM paper [9] |
| Target = -finish_position | ✅ | — | Convenção para coeficiente maior = melhor |
| Causalidade corrida a corrida | ✅ | — | Anti-leakage rule explícita no manifest |
| Alpha tunado via cross-validation | — | ⚠️ | Arquitetura previa tuning; implementado com default 10.0 |
| `grid_position` como covariável no RAPM | — | ⚠️ | Não incluída; modelo é Fixed Effects, não RAPM estrito |
| Correlação construtor > piloto | ✅ | — | Consistente com Snoeks [10] (88% construtor na era híbrida) |
