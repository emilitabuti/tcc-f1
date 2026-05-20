# Plano detalhado de quarta — valores ausentes, outliers e dataset limpo

Data de referência do cronograma: quarta-feira da Semana 1.

Objetivo do dia: fechar uma base historica limpa, auditavel e pronta para feature engineering, garantindo que valores ausentes e outliers sejam tratados antes da modelagem e que as decisoes estejam alinhadas com a arquitetura do TCC.

## Veredito sobre o cronograma de quarta

O cronograma esta bom no nivel macro: os tres blocos previstos fazem sentido para encerrar a preparacao da base antes do feature engineering.

Porem ha um ajuste metodologico importante. Na arquitetura, a ordem natural e:

1. limpeza inicial;
2. tratamento de valores ausentes;
3. tratamento de outliers;
4. encoding;
5. normalizacao;
6. feature engineering.

No cronograma revisado, encoding e normalizacao aparecem na terca, enquanto ausentes e outliers aparecem na quarta. Para a banca, a solucao mais defensavel e documentar a quarta como etapa de auditoria e consolidacao: validar ausentes, aplicar tratamento de outliers nas colunas numericas originais e, se alguma linha/valor mudar, regerar encoding e normalizacao depois.

No estado atual do repositorio, as bases processadas principais nao apresentam valores ausentes, entao a quarta nao deve gastar tempo em imputacao artificial. A prioridade real deve ser:

- provar que nao ha nulos nas bases finais;
- registrar como a imputacao seria aplicada caso aparecam nulos em execucoes futuras;
- implementar o detector de outliers por circuito/ano;
- separar outlier legitimo de outlier espurio;
- gerar relatorio e base limpa final.

## Embasamento bibliografico

| Decisao | Referencias da arquitetura |
|---|---|
| Filtrar era hibrida e manter consistencia regulatoria | [7], [9] |
| Remover nulos em `grid_position` e `finish_position` | Arquitetura, Tratamento dos Dados; coerente com benchmarks de predicao real de resultado [3], [4], [5] |
| Imputar tempos de volta por mediana do circuito/ano | Arquitetura, Valores ausentes; robustez a valores extremos em dados tabulares [3] |
| Imputar composto por moda da corrida | Arquitetura, Valores ausentes; uso de features de pneu/estrategia em F1 [8] |
| Usar KNN para qualifying ausente | Arquitetura, Valores ausentes; preservar amostras raras sem introduzir vazamento |
| Detectar outliers por 3 desvios padrao por circuito | Arquitetura, Outliers; Advanced ML paper [3] |
| Manter outliers legitimos com `safety_car_flag` | Arquitetura, Outliers; eventos de corrida sao parte do dominio, nao erro de sensor |
| Evitar data leakage | Trabalhos de predicao genuina em F1 [2], [3], [4], [5] |
| Preparar base para walk-forward validation | RAPM/time-decay [9], [18] |

## Entradas

Usar como entrada principal:

- `data/processed/historico_dnf_classificado_2018_2025.csv`
- `data/processed/historico_dnf_excluded_2018_2025.csv`
- `data/processed/base_historica_dnf_classificado_2018_2025.csv`
- `data/processed/base_historica_dnf_excluded_2018_2025.csv`

Usar a base classificada para calcular flags e taxas futuras de DNF. Usar a base DNF Excluded para a base principal de treinamento, conforme decisao metodologica ja documentada.

## Saidas esperadas

| Entregavel | Caminho sugerido | Conteudo |
|---|---|---|
| Script de quarta | `src/tratamento_ausentes_outliers.py` | Auditoria de nulos, imputacao condicional, deteccao/remocao de outliers |
| Base historica limpa final | `data/processed/base_historica_final_2018_2025.csv` | Base DNF Excluded apos validacao de ausentes e outliers |
| Historico enriquecido final | `data/processed/historico_final_2018_2025.csv` | Base com FastF1 validada para feature engineering |
| Relatorio da quarta | `data/processed/relatorio_04_ausentes_outliers.txt` | Contagens, regras, outliers mantidos/removidos |
| Documento metodologico | `docs/metodologia_ausentes_outliers.md` | Texto pronto para metodologia |

Se houver pouco tempo, priorizar:

1. relatorio de nulos;
2. detector de outliers por circuito/ano;
3. base final DNF Excluded;
4. metodologia curta;
5. reexecucao do encoding se houver mudanca na base.

## Plano operacional

### 1. Congelar a ordem metodologica

Registrar no documento da quarta:

```text
Embora o cronograma tenha antecipado o encoding, o tratamento de ausentes e outliers foi validado nas variaveis originais antes da modelagem. Quando o tratamento altera linhas ou valores, o encoding deve ser regerado a partir da base limpa final.
```

Criterio de aceite:

- A metodologia explica a ordem real.
- Nenhuma decisao fica parecendo acidental.

### 2. Auditar valores ausentes

Verificar nulos por coluna nas bases DNF Excluded e classificadas.

Colunas criticas:

- `season`
- `round`
- `RaceID`
- `driver_id`
- `constructor_id`
- `grid_position`
- `finish_position`
- `laps`
- `fastf1_avg_lap_time`
- `fastf1_best_lap_time`
- `fastf1_avg_sector1`
- `fastf1_avg_sector2`
- `fastf1_avg_sector3`
- `fastf1_first_compound`
- `fastf1_main_compound`

No estado atual do repositorio, `historico_dnf_excluded_2018_2025.csv`, `base_historica_dnf_excluded_2018_2025.csv` e `historico_encoded_2018_2025.csv` estao sem nulos. Portanto, registrar:

```text
Valores ausentes encontrados na base final: 0.
Imputacao nao aplicada nesta execucao.
Regras de imputacao mantidas no pipeline para execucoes futuras.
```

Criterio de aceite:

- O relatorio lista nulos por coluna.
- Se nulos forem zero, nenhuma imputacao e aplicada.
- Se nulos aparecerem futuramente, as regras abaixo entram automaticamente.

### 3. Imputacao condicional

Implementar as regras, mas executar apenas se houver nulos.

#### 3.1 Tempos de volta e setores

Colunas:

- `fastf1_avg_lap_time`
- `fastf1_best_lap_time`
- `fastf1_avg_sector1`
- `fastf1_avg_sector2`
- `fastf1_avg_sector3`

Regra:

1. imputar pela mediana de `season + race_name`;
2. se ainda faltar, usar mediana de `race_name`;
3. se ainda faltar, usar mediana global da coluna.

Justificativa: a arquitetura define mediana do circuito naquele ano. Os niveis 2 e 3 sao fallback operacional para nao quebrar o pipeline.

#### 3.2 Composto de pneu

Colunas:

- `fastf1_first_compound`
- `fastf1_main_compound`

Regra:

1. imputar pela moda de `season + round`;
2. se ainda faltar, usar moda de `race_name`;
3. se ainda faltar, usar `UNKNOWN`.

#### 3.3 Qualifying

Se a base tiver colunas de qualifying ou setores de qualifying:

1. aplicar `KNNImputer`;
2. ajustar o imputer apenas no conjunto de treino em cada fold futuro;
3. nao usar `finish_position`, `points` ou qualquer dado pos-corrida como variavel auxiliar.

Nesta quarta, se essas colunas ainda nao existem na base, registrar como pendencia controlada:

```text
Colunas de qualifying ainda nao integradas ao dataset final; regra KNN documentada para quando a integracao ocorrer.
```

### 4. Detectar outliers por circuito e ano

Aplicar regra de 3 desvios padrao por grupo `season + race_name`.

Colunas numericas candidatas:

- `fastf1_avg_lap_time`
- `fastf1_best_lap_time`
- `fastf1_avg_sector1`
- `fastf1_avg_sector2`
- `fastf1_avg_sector3`
- `fastf1_max_tyre_life`
- `fastf1_stints_count`
- `fastf1_pit_in_count`
- `fastf1_pit_out_count`
- `laps`

Criar colunas auxiliares:

- `outlier_flag`
- `outlier_columns`
- `outlier_reason`
- `safety_car_flag`
- `outlier_action`

Criterio tecnico:

```text
z = abs(valor - media_do_grupo) / desvio_padrao_do_grupo
outlier se z > 3
```

Grupos com desvio padrao zero ou menos de 5 registros devem ser ignorados para evitar falso positivo.

### 5. Separar outlier legitimo de espurio

Manter outliers legitimos quando houver indicio de evento real de corrida.

Indicios de legitimidade:

- `status` diferente de `Finished` ou `+x Laps` na base classificada;
- pit stops muito acima do padrao da corrida;
- `fastf1_laps_count` muito diferente por evento real;
- corrida com chuva pelo composto `INTERMEDIATE` ou `WET`;
- futura integracao com race control/safety car da OpenF1.

Remover ou marcar como espurio quando:

- tempo de volta ou setor for impossivel fisicamente;
- valor for muito alto por erro de conversao;
- contagem de voltas FastF1 estiver incompatível com `laps` sem explicacao;
- houver `NaN` transformado em numero artificial por erro anterior.

Para esta quarta, se ainda nao houver race control integrado, usar `safety_car_flag = 0` por padrao e registrar a limitacao. Nao inventar safety car.

### 6. Regras de remocao

Nao remover automaticamente todo z-score acima de 3.

Regra recomendada:

- manter outlier legitimo com flag;
- remover outlier espurio apenas se o valor comprometer a base;
- quando houver duvida, manter com flag e relatar.

Isso e mais defensavel porque eventos extremos fazem parte de corridas de F1 e podem ser informativos para modelos de predicao.

### 7. Regenerar base final

Se nenhuma linha for removida:

- copiar base validada para `base_historica_final_2018_2025.csv`;
- copiar historico validado para `historico_final_2018_2025.csv`.

Se linhas forem removidas:

- salvar base final;
- registrar IDs removidos;
- regerar encoding usando a base final, nao a base antiga.

### 8. Validacoes finais

Checklist minimo:

- `RaceID` unico;
- nulos em colunas criticas = 0;
- `finish_position` nao entra como feature;
- `points` nao entra como feature;
- DNF Excluded preservado para treino;
- base classificada preservada para calculo futuro de `driver_dnf_rate` e `constructor_dnf_rate`;
- contagem de linhas antes/depois registrada;
- outliers removidos listados por `RaceID`;
- outliers mantidos listados com flag.

## Divisao sugerida P1/P2

### P1

- Implementar auditoria de nulos.
- Implementar imputacao condicional.
- Gerar primeira versao do relatorio.

### P2

- Implementar detector de outliers.
- Classificar outliers mantidos/removidos.
- Conferir amostras manualmente.

### Juntas

- Validar se a base final mudou.
- Se mudou, rerodar DNF/encoding.
- Fechar `docs/metodologia_ausentes_outliers.md`.

## Cronograma do dia

### 09:00-10:00

Ler relatorios 01, 02 e 03. Confirmar dimensoes atuais:

- base limpa 2018-2025: 3458 linhas;
- DNF Excluded 2018-2025: 2943 linhas;
- historico encoded 2018-2025: 2943 linhas;
- nulos nas bases principais: 0.

### 10:00-12:00

Criar `src/tratamento_ausentes_outliers.py` com:

- leitura das bases;
- validacao de colunas;
- auditoria de nulos;
- funcoes de imputacao condicional;
- log de antes/depois.

### 13:00-15:00

Implementar outliers:

- z-score por `season + race_name`;
- flags por coluna;
- classificacao inicial de legitimo/espurio;
- tabela de outliers para revisao.

### 15:00-16:00

Revisao manual:

- olhar os 20 maiores outliers de tempo de volta;
- olhar os 20 maiores outliers de pit stop/stint;
- decidir manter/remover;
- registrar justificativa.

### 16:00-17:00

Gerar artefatos finais:

- `base_historica_final_2018_2025.csv`;
- `historico_final_2018_2025.csv`;
- `relatorio_04_ausentes_outliers.txt`;
- `metodologia_ausentes_outliers.md`.

### 17:00-18:00

Reexecutar encoding se a base mudou. Caso contrario, registrar que `relatorio_03_encoding.txt` continua valido.

## Riscos e ajustes

### Risco 1: base sem nulos parece que a quarta foi desnecessaria

Resposta: nao foi. A entrega da quarta vira auditoria de qualidade e outliers, que e essencial para defender a base.

### Risco 2: outliers demais removidos

Resposta: remover somente espurios. Corridas com chuva, safety car, pit stops anormais e estrategias extremas devem ser mantidas com flag.

### Risco 3: `safety_car_flag` ainda nao esta disponivel

Resposta: criar a coluna como pendente/zero e documentar que sera preenchida pela integracao OpenF1 race control. Nao preencher manualmente sem fonte.

### Risco 4: KNN de qualifying sem coluna de qualifying

Resposta: documentar a regra e marcar como pendencia da integracao FastF1 qualifying. Nao aplicar KNN em variaveis erradas so para cumprir o cronograma.

## Criterio de pronto

A quarta esta concluida quando:

- existe relatorio de ausentes/outliers;
- a base final nao tem nulos criticos;
- outliers foram analisados por circuito/ano;
- qualquer remocao esta justificada por `RaceID`;
- a base final esta pronta para RAPM e feature engineering de quinta;
- as decisoes citam as referencias da arquitetura.

