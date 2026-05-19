# Plano detalhado de terça — limpeza, DNF, encoding e normalização

Data de referência do cronograma: terça-feira da Semana 1.

Objetivo do dia: transformar os dados brutos já extraídos em uma base preliminar padronizada, deduplicada e pronta para receber imputação, outliers e feature engineering nos próximos blocos. A normalização fica neste dia por decisão do projeto, mas deve ser implementada como pipeline reproduzível para evitar vazamento de dados.

---

## Base de embasamento

Este plano foi detalhado a partir de quatro tipos de fonte:

1. `docs/ArquiteturaProposta.pdf`: define as escolhas metodológicas principais da Fase 1, incluindo limpeza, DNF, encoding, normalização, features e referências bibliográficas.
2. `docs/Cronograma_Revisado.pdf`: define o que precisa ser executado especificamente na terça-feira.
3. `dados_necessarios.txt`: mapeia campos, fontes e features, indicando quais decisões vêm de TabNet, RAPM, Koopman, Pit Stop paper, OpenF1/FastF1 docs e demais referências já levantadas no projeto.
4. Estado atual do repositório: nomes reais dos arquivos em `data/raw/`, colunas disponíveis e lacunas vistas em `data/raw/dados_ausentes.txt`.

Tabela de rastreabilidade:

| Decisão no plano | Embasamento |
|---|---|
| Remover registros com `grid_position` ou `finish_position` nulos | `ArquiteturaProposta.pdf`, seção Tratamento dos Dados; `Cronograma_Revisado.pdf`, bloco de terça |
| Criar `RaceID = piloto + temporada + round` | `ArquiteturaProposta.pdf`, seção Tratamento dos Dados; `Cronograma_Revisado.pdf`, bloco de terça |
| Remover duplicatas por `RaceID` | `ArquiteturaProposta.pdf`; `Cronograma_Revisado.pdf` |
| Filtrar era híbrida ou justificar escopo temporal | `ArquiteturaProposta.pdf` cita 2014+; estado atual do repositório mostra cobertura operacional 2018–2025 |
| Usar `grid_position` como feature e `finish_position` como target | `dados_necessarios.txt`, seções Jolpica `/results/` e resumo de features finais; referências TabNet, Koopman, Barra e demais papers já mapeados |
| Preservar `points` apenas para auditoria, sem usar como feature principal | `dados_necessarios.txt`, seção Jolpica `/results/`, marca `points` como risco de data leakage |
| Variante `DNF Excluded` | `Cronograma_Revisado.pdf`, bloco de terça; `ArquiteturaProposta.pdf`, referências ao benchmark RAPM |
| Separar DNFs de piloto vs carro | `Cronograma_Revisado.pdf`, bloco de terça; `ArquiteturaProposta.pdf`, feature engineering de `driver_dnf_rate` e `constructor_dnf_rate` |
| Manter base com DNF para calcular taxas futuras | Derivado da própria arquitetura, que usa `driver_dnf_rate` e `constructor_dnf_rate`; se DNFs forem apagados antes, essas features perdem base |
| One-hot para circuito e construtor | `ArquiteturaProposta.pdf`, seção Encoding; `Cronograma_Revisado.pdf`, bloco de terça |
| Não usar one-hot de piloto no dataset principal | `ArquiteturaProposta.pdf`, seção Encoding, que define piloto como coeficiente RAPM numérico |
| Label/ordinal encoding para composto de pneu | `ArquiteturaProposta.pdf`, seção Encoding; `Cronograma_Revisado.pdf`, bloco de terça |
| Incluir `INTERMEDIATE`, `WET` e `UNKNOWN` no mapa de pneus | Decisão operacional baseada nos campos reais FastF1/OpenF1 descritos em `dados_necessarios.txt`; complementa a arquitetura, que cita apenas Soft/Medium/Hard |
| Z-score para variáveis contínuas | `ArquiteturaProposta.pdf`, seção Normalização; `Cronograma_Revisado.pdf`, bloco de terça |
| MinMaxScaler para `grid_position` e `laps` | `ArquiteturaProposta.pdf`, seção Normalização; `Cronograma_Revisado.pdf`, bloco de terça |
| Separar normalização por pipeline para evitar leakage | Boas práticas de avaliação temporal e coerência com walk-forward/time-decay da arquitetura; não é uma citação literal do cronograma, é salvaguarda metodológica |
| Gerar relatório de execução e artefatos reproduzíveis | `Cronograma_Revisado.pdf` pede reprodutibilidade e documentação; detalhe operacional meu para deixar a execução auditável |
| Usar nomes como `data/processed/base_historica_preliminar.csv` | Decisão operacional para organizar entregáveis; não vem diretamente de paper |

Itens que são extrapolações operacionais, não afirmações bibliográficas:

- nomes exatos dos arquivos gerados;
- formato do relatório `preprocess_report.md`;
- nomes dos artefatos `.joblib`;
- classificação inicial detalhada de status de DNF além dos exemplos citados no cronograma;
- recomendação de manter `grid_position = 0` com flag, que precisa ser validada empiricamente no dataset;
- regra de `UNKNOWN = 0` para pneus ausentes.

Esses pontos devem ser tratados como proposta de implementação e documentados em `docs/decisoes_preprocessamento.md` quando forem aceitos pelo grupo.

---

## Entregáveis esperados

Ao final da terça, devem existir:

| Entregável | Caminho sugerido | Conteúdo |
|---|---|---|
| Script de preprocessamento | `src/preprocess_dataset.py` | Leitura dos CSVs brutos, limpeza estrutural, DNF, encoding e normalização |
| Dataset base limpo | `data/processed/base_historica_preliminar.csv` | Dados por piloto/corrida após limpeza estrutural |
| Dataset modelável preliminar | `data/processed/base_modelavel_preliminar.csv` | Dataset após encoding e normalização |
| Relatório de execução | `data/processed/preprocess_report.md` | Quantidade de linhas antes/depois, nulos, duplicatas, DNFs e decisões |
| Artefatos de preprocessamento | `models/preprocessing/` | Encoders/scalers salvos ou configuração equivalente versionada |
| Documento metodológico | `docs/decisoes_preprocessamento.md` | Regras adotadas para DNF, encoding, normalização e escopo temporal |

Se não houver tempo para todos, priorizar nesta ordem:

1. `src/preprocess_dataset.py`
2. `data/processed/base_historica_preliminar.csv`
3. `docs/decisoes_preprocessamento.md`
4. `data/processed/base_modelavel_preliminar.csv`
5. `models/preprocessing/`

---

## Ordem de execução

### 1. Confirmar escopo temporal

Decisão necessária antes de limpar:

- O cronograma original fala em era híbrida a partir de 2014.
- O estado atual da base está majoritariamente em 2018–2025.

Decisão recomendada para terça:

> Usar 2018–2025 como escopo operacional da Fase 1, porque é o período já coberto pelas extrações atuais de Ergast/Jolpica, FastF1 e OpenF1. Registrar que 2014–2017 pode ser expandido depois, se necessário para a versão final do TCC.

Registrar em `docs/decisoes_preprocessamento.md`:

```text
Escopo temporal operacional: 2018–2025.
Justificativa: cobertura consistente entre fontes disponíveis no repositório.
Observação: a arquitetura cita era híbrida 2014+, mas a Fase 1 usa 2018+ por disponibilidade e consistência dos dados extraídos.
```

Critério de aceite:

- Dataset final da terça contém apenas `season >= 2018`.
- Caso 2014–2017 não exista, isso aparece como decisão metodológica, não como falha silenciosa.

---

### 2. Definir unidade da base

A unidade da base preliminar deve ser:

> uma linha por piloto em uma corrida.

Fonte principal:

- `data/raw/ergast_2018_2024.csv`
- `data/raw/ergast_2025_results.csv`

Colunas mínimas esperadas:

| Coluna | Uso |
|---|---|
| `season` | ordenação temporal e split |
| `round` | ordenação dentro da temporada |
| `race_name` | conferência humana |
| `driver_id` | chave do piloto |
| `constructor_id` | chave do construtor |
| `grid_position` | feature principal |
| `finish_position` | target |
| `status` | DNF e confiabilidade |
| `points` | auditoria, não usar como feature do modelo principal |
| `laps` | suporte para DNF e validação |

Critério de aceite:

- As colunas mínimas existem.
- `points` fica preservado para auditoria, mas marcado como `excluded_feature` para evitar data leakage.

---

### 3. Criar `RaceID`

Criar uma chave única por piloto/corrida:

```text
RaceID = season + "_" + round + "_" + driver_id
```

Exemplo:

```text
2025_1_norris
```

Motivo:

- `season + round` identifica a corrida dentro do calendário.
- `driver_id` diferencia cada piloto.
- `race_name` não deve entrar no ID porque é texto livre e pode variar entre fontes.

Validações:

- Não pode haver `RaceID` nulo.
- `RaceID` deve ser único.
- Duplicatas por `RaceID` devem ser removidas e registradas no relatório.

Critério de aceite:

```text
duplicatas_RaceID = 0
RaceID_nulos = 0
```

---

### 4. Limpeza estrutural

Aplicar as regras abaixo antes de qualquer encoding ou normalização.

#### 4.1 Remover registros sem grid ou target

Remover linhas com:

- `grid_position` nulo
- `finish_position` nulo
- `driver_id` nulo
- `constructor_id` nulo
- `season` nulo
- `round` nulo

Registrar no relatório:

```text
linhas_removidas_grid_nulo = X
linhas_removidas_finish_position_nulo = X
linhas_removidas_chaves_nulas = X
```

Observação:

- `grid_position = 0` precisa ser tratado com cuidado. Em Ergast/Jolpica, `0` normalmente indica largada do pit lane ou ausência de posição formal. Não remover automaticamente antes de documentar.

Decisão recomendada:

- Manter `grid_position = 0`.
- Criar flag `grid_position_zero_flag = 1`.
- Para modelos que exigirem escala ordinal limpa, transformar `grid_position_model = max(grid_position, 20)` ou decidir depois conforme análise.

#### 4.2 Padronizar tipos

Converter:

| Coluna | Tipo |
|---|---|
| `season` | inteiro |
| `round` | inteiro |
| `grid_position` | inteiro ou float se houver nulo temporário |
| `finish_position` | inteiro |
| `laps` | inteiro |
| `driver_id` | string |
| `constructor_id` | string |
| `status` | string |

#### 4.3 Ordenar a base

Ordenar por:

```text
season, round, finish_position
```

Isso facilita conferência, cálculo de features acumuladas e validação visual.

---

### 5. Tratamento e documentação de DNFs

O cronograma define a variante principal como:

> DNF Excluded, alinhada ao benchmark RAPM.

Mas os DNFs ainda precisam ser preservados para calcular features futuras:

- `driver_dnf_rate`
- `constructor_dnf_rate`
- confiabilidade do carro
- perfil de agressividade do piloto

Portanto, usar duas bases intermediárias:

| Base | Conteúdo | Uso |
|---|---|---|
| `base_com_dnf` | mantém todos os registros válidos | cálculo de taxas de DNF |
| `base_model_target` | exclui DNFs conforme regra escolhida | treino do target de posição final |

### 5.1 Criar flags de DNF

Criar:

| Coluna | Regra |
|---|---|
| `dnf_flag` | `1` se `status` não indica conclusão normal |
| `dnf_driver_flag` | `1` para acidente/erro/incidente do piloto |
| `dnf_car_flag` | `1` para falha mecânica/sistema do carro |
| `dnf_other_flag` | `1` para casos ambíguos |

### 5.2 Status considerados conclusão normal

Classificar como não DNF:

```text
Finished
+1 Lap
+2 Laps
+3 Laps
+4 Laps
+5 Laps
+6 Laps
+7 Laps
+8 Laps
+9 Laps
```

Regra prática:

- `status == "Finished"` -> não DNF
- `status` começa com `"+"` e termina com `"Lap"` ou `"Laps"` -> não DNF

### 5.3 Status de DNF do piloto

Classificar como `dnf_driver_flag = 1` quando `status` contiver:

```text
Accident
Collision
Collision damage
Spun off
Damage
Puncture
Wheel nut
Withdrew
Disqualified
Excluded
Retired
```

Observação:

- `Puncture`, `Damage` e `Wheel nut` podem ter origem ambígua. Se o grupo preferir ser conservador, mover para `dnf_other_flag`.
- `Disqualified` e `Excluded` não são falha de pilotagem necessariamente, mas afetam resultado; manter separado se houver tempo.

### 5.4 Status de DNF mecânico/carro

Classificar como `dnf_car_flag = 1` quando `status` contiver:

```text
Engine
Gearbox
Transmission
Clutch
Hydraulics
Electrical
Electronics
ERS
Power Unit
Turbo
MGU-K
MGU-H
Battery
Brakes
Suspension
Wheel
Driveshaft
Fuel
Oil leak
Water leak
Cooling system
Radiator
Exhaust
Throttle
Steering
```

### 5.5 Casos ambíguos

Qualquer status não classificado como conclusão normal, DNF de piloto ou DNF de carro entra como:

```text
dnf_flag = 1
dnf_other_flag = 1
```

O relatório deve listar os status ambíguos encontrados:

```text
status_ambiguos:
- Status A: n ocorrências
- Status B: n ocorrências
```

### 5.6 Exclusão para base de treino

Gerar `base_model_target` removendo:

```text
dnf_flag == 1
```

Mas manter `base_com_dnf` salva ou reprodutível para feature engineering.

Critério de aceite:

- Toda linha tem exatamente uma classificação: normal, DNF piloto, DNF carro ou DNF outro.
- Quantidade de DNFs removidos aparece no relatório.
- A decisão “DNF Excluded” aparece em `docs/decisoes_preprocessamento.md`.

---

### 6. Encoding

Encoding deve ser aplicado depois da limpeza estrutural e da classificação de DNF.

Importante:

- O encoding precisa ser reprodutível.
- Categorias desconhecidas no futuro não podem quebrar o pipeline.

### 6.1 One-hot encoding

Aplicar one-hot em:

```text
circuit_id ou race_name
constructor_id
```

Preferência:

1. Usar `circuit_id` se já houver join confiável com circuitos.
2. Usar `race_name` temporariamente se `circuit_id` ainda não estiver integrado.

Configuração recomendada:

```python
OneHotEncoder(handle_unknown="ignore", sparse_output=False)
```

Observação:

- `driver_id` não deve virar one-hot no dataset principal se o RAPM for usado como feature numérica depois.
- Para terça, manter `driver_id` como chave categórica bruta e não como feature final.

### 6.2 Encoding ordinal de pneu

Para `tire_compound_start`, quando a coluna estiver integrada:

| Composto | Valor |
|---|---:|
| `SOFT` | 5 |
| `MEDIUM` | 4 |
| `HARD` | 3 |
| `INTERMEDIATE` | 2 |
| `WET` | 1 |
| `UNKNOWN` ou nulo | 0 |

Justificativa:

- O cronograma cita Soft > Medium > Hard.
- Intermediário e chuva precisam entrar para não quebrar corridas molhadas.
- `UNKNOWN = 0` evita descarte indevido antes da imputação definitiva.

Se `tire_compound_start` ainda não estiver pronto na terça:

- Deixar a função de encoding preparada.
- Registrar no relatório que a coluna será preenchida na etapa de feature engineering.

Critério de aceite:

- One-hot com `handle_unknown="ignore"`.
- Mapa de pneu documentado.
- Nenhuma coluna categórica usada diretamente pelo modelo sem encoding.

---

### 7. Normalização

Por decisão do projeto, a normalização fica no plano de terça.

Regra metodológica importante:

> Mesmo normalizando hoje, os scalers não devem ser ajustados na base inteira quando houver avaliação temporal. Para evitar vazamento, implementar a normalização como pipeline e deixar claro que o `fit` final será feito apenas no conjunto de treino em cada split.

### 7.1 Colunas para Z-score

Aplicar `StandardScaler` nas variáveis numéricas contínuas disponíveis na terça:

```text
laps
```

E preparar para as futuras:

```text
recent_form_5
recent_form_3
driver_experience
driver_wins_total
driver_coef_rapm
driver_dnf_rate
constructor_coef_rapm
constructor_dnf_rate
constructor_wins_total
driver_constructor_synergy
track_complexity
altitude
weather_impact_factor
avg_pit_stops_circuit
season_factor
```

### 7.2 Colunas para MinMaxScaler

Aplicar `MinMaxScaler` em:

```text
grid_position
laps
```

Observação:

- O cronograma cita `GridPosition` e `Laps`.
- Se `laps` também estiver no Z-score, escolher uma versão por modelo:
  - `laps_z` para Ridge/baselines lineares
  - `laps_minmax` se necessário para redes/modelos sensíveis à escala

### 7.3 Colunas que não devem ser normalizadas

Não normalizar:

```text
finish_position
points
season
round
RaceID
driver_id
constructor_id
status
dnf_flag
dnf_driver_flag
dnf_car_flag
dnf_other_flag
one-hot columns
```

Motivos:

- `finish_position` é target.
- `points` não é feature principal por risco de leakage.
- flags e one-hot já estão em escala binária.
- chaves são identificadores, não medidas.

### 7.4 Artefatos

Salvar, se possível:

```text
models/preprocessing/onehot_encoder.joblib
models/preprocessing/standard_scaler.joblib
models/preprocessing/minmax_scaler.joblib
models/preprocessing/preprocessing_config.json
```

Se ainda não for salvar objetos, ao menos salvar `preprocessing_config.json` com:

- colunas usadas no one-hot
- mapa ordinal de pneus
- colunas padronizadas por Z-score
- colunas normalizadas por MinMax
- data de geração
- escopo temporal

Critério de aceite:

- O dataset modelável tem colunas normalizadas com nomes explícitos, por exemplo `grid_position_minmax`, `laps_z`, `laps_minmax`.
- As colunas originais são preservadas na base preliminar.
- O relatório informa quais scalers foram aplicados.

---

### 8. Evitar data leakage

Checagens obrigatórias:

| Risco | Regra |
|---|---|
| `points` como feature | Não usar no modelo principal |
| Normalizar com dados futuros | Pipeline deve permitir `fit` apenas no treino |
| Features acumuladas olhando o futuro | Calcular apenas com corridas anteriores |
| Encoding quebrar em 2025/2026 | Usar `handle_unknown="ignore"` |
| DNF excluído antes de calcular taxa de DNF | Calcular taxas futuras usando base com DNF preservada |

Para terça, registrar no relatório:

```text
points_preservado_para_auditoria = sim
points_usado_como_feature = nao
dnf_preservado_para_feature_engineering = sim
dnf_excluido_da_base_model_target = sim
```

---

## Checklist operacional

### Preparação

- [ ] Criar diretórios `data/processed/` e `models/preprocessing/`, se não existirem.
- [ ] Criar `src/preprocess_dataset.py`.
- [ ] Carregar `ergast_2018_2024.csv` e `ergast_2025_results.csv`.
- [ ] Concatenar resultados históricos e 2025.
- [ ] Filtrar `season >= 2018`.

### Limpeza estrutural

- [ ] Verificar colunas mínimas.
- [ ] Padronizar tipos.
- [ ] Criar `RaceID`.
- [ ] Remover duplicatas por `RaceID`.
- [ ] Remover linhas com chaves essenciais nulas.
- [ ] Remover linhas com `grid_position` ou `finish_position` nulos.
- [ ] Criar `grid_position_zero_flag`.
- [ ] Ordenar por `season`, `round`, `finish_position`.

### DNF

- [ ] Criar `dnf_flag`.
- [ ] Criar `dnf_driver_flag`.
- [ ] Criar `dnf_car_flag`.
- [ ] Criar `dnf_other_flag`.
- [ ] Listar status ambíguos.
- [ ] Salvar base com DNF preservado ou garantir reprodutibilidade.
- [ ] Gerar base de target com DNFs excluídos.
- [ ] Documentar a decisão DNF Excluded.

### Encoding

- [ ] Definir coluna de circuito usada no one-hot: `circuit_id` ou `race_name`.
- [ ] Aplicar one-hot em circuito.
- [ ] Aplicar one-hot em `constructor_id`.
- [ ] Preparar mapa ordinal de pneus.
- [ ] Não aplicar one-hot em `driver_id` para o dataset principal.

### Normalização

- [ ] Criar `grid_position_minmax`.
- [ ] Criar `laps_minmax`.
- [ ] Criar `laps_z`.
- [ ] Preservar colunas originais.
- [ ] Salvar scalers/configuração.
- [ ] Registrar no relatório que o fit final por split temporal deve ocorrer apenas no treino.

### Relatórios

- [ ] Gerar `data/processed/preprocess_report.md`.
- [ ] Criar ou atualizar `docs/decisoes_preprocessamento.md`.
- [ ] Registrar total de linhas brutas.
- [ ] Registrar total de linhas após limpeza.
- [ ] Registrar duplicatas removidas.
- [ ] Registrar DNFs totais e por tipo.
- [ ] Registrar linhas removidas por DNF da base de treino.
- [ ] Registrar colunas de encoding.
- [ ] Registrar colunas normalizadas.

---

## Critério para dizer que terça está concluída

A terça pode ser considerada concluída quando:

```text
1. Existe uma base preliminar em data/processed/.
2. RaceID é único e sem nulos.
3. A regra de DNF está implementada e documentada.
4. Existe uma base de modelagem sem DNFs, se essa for a variante escolhida.
5. Encoding de circuito/construtor está definido ou implementado.
6. Normalização está implementada com nomes de colunas explícitos.
7. O relatório mostra quantas linhas foram removidas e por quê.
8. Decisões metodológicas estão registradas em docs/decisoes_preprocessamento.md.
```

Se algum item não for concluído, registrar como pendência objetiva:

```text
Pendência:
- O que falta:
- Por que não foi feito:
- Arquivo afetado:
- Próxima ação:
```

---

## Observações para a quarta-feira

Mesmo com a normalização feita hoje, a quarta-feira ainda deve cuidar de:

- imputação dos tempos de volta;
- imputação/moda do composto de pneu;
- imputação de qualifying;
- tratamento de outliers por circuito;
- criação de `safety_car_flag` a partir de `TrackStatus` e OpenF1 race control;
- revisão da normalização após imputação, se os valores imputados alterarem a distribuição.

Se a quarta alterar valores numéricos usados em scaler, regenerar `base_modelavel_preliminar.csv` e os artefatos de preprocessamento.
