    # 02 - Limpeza e Tratamento de DNF

## Contexto

Após a coleta, a base bruta contém registros de todo piloto que participou de cada corrida, incluindo aqueles que abandonaram antes do fim. O problema é que a posição final de um piloto que abandonou não é comparável à de um piloto que completou a corrida: um piloto que largou em 3º e abandona por falha mecânica na volta 10 pode ser registrado nas últimas posições, mas esse resultado não mede diretamente seu desempenho competitivo.

O objetivo desta etapa é: (1) remover registros inválidos da base bruta, e (2) decidir o que fazer com os registros de abandono.

---

## Fundamentação bibliográfica

| Decisão | Referência |
|---|---|
| Variante DNF Excluded como abordagem padrão | Henderson et al. [9] - RAPM paper (MAE benchmark de 2.3 adotado com DNF Excluded) |
| Classificação de DNF em categorias (piloto/mecânico) | Ruan et al. [2] - RF+SHAP paper usa `driver_dnf_rate` e `constructor_dnf_rate` como features separadas |
| Documentar DNF como limitação metodológica | Arquitetura proposta, seção 2 - "Outliers legítimos como falhas mecânicas: manter com flag binária" |

A arquitetura menciona o tratamento de outliers legítimos com flag binária, como `safety_car_flag = 1`. Para DNFs, a decisão foi diferente: exclusão completa do dataset de modelagem, seguindo a variante **DNF Excluded** de Henderson et al. [9].

Essa diferença é metodologicamente importante. Safety Car é um evento real de corrida, mas a linha do piloto ainda possui uma posição oficial classificável. Já um DNF ou uma desclassificação mistura desempenho com abandono, falha, acidente, punição ou ausência de classificação válida. Por isso, DNF não foi mantido como flag em `X`.

No pipeline final, `safety_car_flag` também não entra como feature preditiva, porque é informação pós-corrida. Ela permanece como auditoria e foi substituída em `X` por `incident_rate_hist_norm`, uma taxa histórica causal de incidentes por circuito.

---

## Implementação

### Script: `src/tratamento_dnf.py`

O script recebe a base limpa da Etapa 01 e aplica dois passos: classificação e exclusão.

### Passo 1 - Critérios de registro inválido (Etapa 01)

Os seguintes critérios removem registros na etapa de limpeza, antes do tratamento de DNF:

| Critério | Coluna | Ação |
|---|---|---|
| `grid_position` nulo | `grid_position` | Remover |
| `finish_position` nulo | `finish_position` | Remover |
| `driver_id` nulo | `driver_id` | Remover |
| `constructor_id` nulo | `constructor_id` | Remover |
| `season` ou `round` nulo | `season`, `round` | Remover |
| RaceID duplicado | `driver_id + season + round` | Manter primeiro, remover demais |

Resultado da Etapa 01: **0 registros removidos** por nulos. A base Ergast/Jolpica 2018-2025 está completa nesses campos. Também foram encontradas **0 duplicatas** por `RaceID`.

Observação: existem **45 registros com `grid_position = 0`**, mas isso não é nulo. Esses casos são mantidos e marcados por `grid_position_zero_flag`, pois podem representar largada do pit lane, punição, DNS ou ausência de posição formal.

### Passo 2 - Classificação de DNF

A função `classificar_dnf()` analisa o campo `status` e atribui uma de quatro categorias:

**`classificado`** - piloto terminou ou foi oficialmente classificado:

- `"Finished"` - completou a prova no mesmo número de voltas do vencedor.
- `"Lapped"` - foi oficialmente classificado, mas com volta(s) a menos.
- Padrão `"+N Lap(s)"` - foi oficialmente classificado com N volta(s) a menos.

**`dnf_piloto`** - abandono por incidente associado ao piloto (7 palavras-chave):

```text
accident, collision, spun off, spun-off, spin, crash, damage
```

**`dnf_carro`** - abandono por falha mecânica/técnica do carro (41 entradas no script):

```text
engine, gearbox, transmission, clutch, hydraulics, electrical, electronics,
ers, power unit, power loss, brakes, brake, suspension, steering, radiator,
oil, oil leak, water pressure, water leak, water pump, cooling system,
fuel, fuel pressure, fuel pump, fuel leak, out of fuel, turbo, exhaust,
mechanical, overheating, puncture, tyre, wheel, wheel nut, driveshaft,
differential, battery, undertray, front wing, rear wing, vibrations
```

**`dnf_outros`** - todos os demais casos reconhecidos ou não reconhecidos:

```text
did not start, dns, withdrew, withdrawn, illness,
excluded, disqualified, retired, not classified
```

A lógica de prioridade é sequencial: primeiro verifica `classificado`, depois `dnf_piloto`, depois `dnf_carro` e, por último, `dnf_outros`. Status vazio ou não reconhecido cai em `dnf_outros`.

Na base atual, apenas um status caiu em `dnf_outros` por fallback sem keyword explícita: `"Debris"` (1 registro). A classificação ainda é adequada, pois o registro não representa uma posição final comparável a uma corrida concluída.

### Passo 3 - Exclusão (DNF Excluded)

Apenas registros com `dnf_categoria == "classificado"` permanecem no dataset de modelagem. Os demais são removidos. A base classificada completa é salva separadamente para rastreabilidade e para calcular `driver_dnf_rate` e `constructor_dnf_rate` na Etapa 11.

### Sobre pilotos desclassificados (`Disqualified`)

A palavra `"disqualified"` está na lista de `DNF_OUTROS_KEYWORDS`. Portanto, pilotos desclassificados recebem `dnf_categoria = "dnf_outros"` e são excluídos do dataset de modelagem.

Isso tem implicação prática confirmada durante a validação: na temporada 2025, Hamilton, Leclerc e Gasly foram desclassificados após o GP da China, e Norris e Piastri após Las Vegas. Esses pilotos não aparecem no fold 2025 não por falha de extração, mas por decisão metodológica: o modelo prediz `finish_position` oficial para pilotos classificados, e desclassificados não possuem uma posição final válida para o objetivo de regressão adotado.

---

## Resultados obtidos

Do `relatorio_02_tratamento_dnf.txt` (base 2018-2025):

| Categoria | Registros | % do total |
|---|---:|---:|
| `classificado` | 2.943 | 85,1% |
| `dnf_carro` | 174 | 5,0% |
| `dnf_piloto` | 147 | 4,3% |
| `dnf_outros` | 194 | 5,6% |
| **Total bruto** | **3.458** | **100%** |

**515 registros removidos** da base de modelagem (14,9% do total bruto).

A diferença de `dnf_outros` entre 2018-2024 (134) e 2018-2025 (194) representa 60 registros a mais em 2025. Parte deles são as desclassificações mencionadas acima.

Base de modelagem resultante: **2.943 linhas**, confirmada nas etapas subsequentes (`dataset_modelagem_2018_2025`, `dataset_modelagem_X_2018_2025` e `dataset_modelagem_y_2018_2025`).

### Distribuição por temporada

| Temporada | Classificado | DNF carro | DNF outros | DNF piloto |
|---:|---:|---:|---:|---:|
| 2018 | 335 | 49 | 5 | 31 |
| 2019 | 360 | 29 | 5 | 26 |
| 2020 | 283 | 30 | 4 | 23 |
| 2021 | 381 | 24 | 5 | 30 |
| 2022 | 366 | 41 | 1 | 32 |
| 2023 | 374 | 1 | 60 | 5 |
| 2024 | 425 | 0 | 54 | 0 |
| 2025 | 419 | 0 | 60 | 0 |

Essa tabela revela uma limitação importante da fonte: em 2024 e 2025, muitos abandonos aparecem como `"Retired"` genérico, sem detalhamento suficiente para separar piloto vs. carro. Isso não invalida a exclusão DNF, mas enfraquece a granularidade das features `driver_dnf_rate` e `constructor_dnf_rate` para temporadas recentes.

---

## Avaliação crítica

### Por que excluir DNFs e não marcar com flag?

Existem duas abordagens comuns:

- **DNF Included com flag**: mantém todos os registros, adiciona `is_dnf = 1` como feature. O modelo aprende uma mistura de corridas completas, abandonos e punições.
- **DNF Excluded**: remove abandonos, treina apenas em corridas classificadas/concluídas.

A escolha por DNF Excluded tem três justificativas:

1. **Target incoerente**: a posição final de um piloto que abandonou na volta 5 não mede seu desempenho; mede quando ele parou.
2. **Alinhamento com o benchmark**: Henderson et al. [9] usa DNF Excluded e reporta MAE de 2.3, que é uma referência comparativa do TCC.
3. **Problema diferente**: previsão de DNF é um problema de classificação separado. Misturá-lo com previsão de posição final contaminaria os dois problemas.

### Viés de sobrevivência introduzido

A exclusão de DNFs cria um viés de sobrevivência documentável: o modelo aprende com a distribuição de pilotos que completaram ou foram classificados na corrida.

Na prática, foram removidos 515 de 3.458 registros (14,9%). Em 2025, o dataset de modelagem mantém 419 linhas classificadas de 479 registros brutos. Logo, o modelo prediz a posição final condicionada a "o piloto foi classificado", não o resultado incondicional de todos os inscritos/largadores.

Esse viés pode ser relevante em temporadas com alta taxa de DNF mecânico. Em 2026, com nova era regulatória de chassi, aerodinâmica e unidade de potência, pode haver instabilidade mecânica maior que a observada no histórico 2018-2025. Nesse caso, corridas com abandonos em novo contexto regulatório podem ocorrer com frequência diferente da aprendida pelo modelo. A limitação é metodológica e deve ser declarada; não é erro de implementação.

### Sobre `dnf_outros` como categoria heterogênea

`dnf_outros` agrupa casos muito diferentes:

- `"Retired"` - abandono genérico sem causa especificada.
- `"Disqualified"` - penalidade pós-corrida.
- `"Did not start"` - piloto inscrito mas não largou.
- `"Illness"` - causa médica.
- `"Debris"` - caso não reconhecido explicitamente, classificado por fallback.

Todos são excluídos da modelagem. Porém, `driver_dnf_rate` e `constructor_dnf_rate` calculadas na Etapa 11 usam apenas `dnf_driver_flag` (piloto) e `dnf_car_flag` (mecânico); `dnf_outros` não entra no numerador dessas taxas.

Isso é metodologicamente correto: uma desclassificação ou um DNS não mede agressividade do piloto nem confiabilidade mecânica do carro. O ponto de atenção é que `"Retired"` genérico pode esconder falhas mecânicas ou incidentes do piloto. Como a causa não está disponível no status, a opção conservadora foi não atribuir esse caso nem ao piloto nem ao carro.

---

## Experimento de ablação das taxas de DNF

Para verificar se `driver_dnf_rate` e `constructor_dnf_rate` ajudam ou prejudicam o desempenho final, foi criado o experimento:

- Script: `src/experimento/experimento_dnf_rates_ablation.py`
- Métricas: `reports/modelagem/experimento_dnf_rates_metricas.csv`
- Relatório: `reports/modelagem/experimento_dnf_rates_relatorio.md`

### Desenho do teste

O teste usa o dataset final de modelagem e o XGBoost já tunado. A validação é walk-forward:

- treino até 2022 -> validação 2023
- treino até 2023 -> validação 2024
- treino até 2024 -> validação 2025

Foram comparados quatro cenários:

| Cenário | Features |
|---|---|
| A - `base_15_features` | Modelo final atual, com as duas taxas DNF |
| B - `sem_driver_dnf_rate` | Remove apenas `driver_dnf_rate` |
| C - `sem_constructor_dnf_rate` | Remove apenas `constructor_dnf_rate` |
| D - `sem_duas_dnf_rates` | Remove as duas taxas |

### Resultado médio

| Cenário | Nº features | MAE médio | Delta MAE vs. base | Kendall tau médio | Top-3 médio |
|---|---:|---:|---:|---:|---:|
| B - `sem_driver_dnf_rate` | 14 | 2.322897 | -0.023486 | 0.652198 | 0.241162 |
| D - `sem_duas_dnf_rates` | 13 | 2.335588 | -0.010796 | 0.647944 | 0.241162 |
| C - `sem_constructor_dnf_rate` | 14 | 2.335602 | -0.010781 | 0.649703 | 0.227273 |
| A - `base_15_features` | 15 | 2.346383 | 0.000000 | 0.652334 | 0.241162 |

### Interpretação

O melhor MAE médio apareceu no cenário sem `driver_dnf_rate`, com melhora de aproximadamente 0,023 posição em relação à base. A diferença é pequena e não muda a conclusão metodológica principal: as taxas DNF são causais e defensáveis, pois usam apenas histórico anterior e são calculadas a partir da base DNF classificada completa.

Entretanto, o teste mostra que o sinal empírico dessas features é fraco para o problema atual. Isso é esperado porque o dataset de modelagem segue DNF Excluded: depois que os abandonos são removidos, `driver_dnf_rate` e `constructor_dnf_rate` tentam explicar a posição final apenas entre pilotos que terminaram/classificaram. Assim, as taxas podem ser mais úteis em um modelo separado de risco de DNF do que no regressor de `finish_position`.

Decisão documentada: manter as duas features no pipeline final é aceitável por alinhamento com Ruan et al. [2] e por coerência causal, mas `driver_dnf_rate` fica registrada como candidata a ablação futura caso o objetivo seja otimizar exclusivamente MAE.

---

## Convergência com a literatura

| Aspecto | Alinhado | Divergente | Observação |
|---|---|---|---|
| Variante DNF Excluded | Sim | - | Mesma abordagem de Henderson et al. [9] |
| Classificação em piloto/mecânico/outro | Sim | - | Necessária para calcular `driver_dnf_rate` e `constructor_dnf_rate` de Ruan et al. [2] |
| `Lapped` como classificado | Sim | - | Correto: é posição oficial, apenas com volta(s) a menos |
| `Disqualified` como `dnf_outros` e excluído | Sim | - | Correto para o objetivo de regressão adotado |
| `dnf_outros` fora das taxas piloto/carro | Sim | - | Evita atribuir DSQ/DNS/causa indefinida a piloto ou construtor |
| Documentar viés de sobrevivência | Sim | - | Limitação explicitada e quantificada |
| Ablação empírica das taxas DNF | Sim | - | Teste adicional indica sinal fraco, mas não invalida a decisão metodológica |
