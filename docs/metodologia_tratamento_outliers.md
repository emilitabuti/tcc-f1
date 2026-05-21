# Tratamento de outliers

## Objetivo

Identificar e tratar valores extremos nas variáveis de tempo de volta e setores, com critério
metodológico que preserve eventos reais de corrida e remova apenas erros técnicos.

## Critério de detecção

Valores acima de 3 desvios padrão da média **por circuito** (não global).

Referência: mesmo critério do paper Barra et al. — Advanced ML for F1 (ref. [3]).

Colunas avaliadas:

- `fastf1_avg_lap_time`
- `fastf1_best_lap_time`
- `fastf1_avg_sector1`
- `fastf1_avg_sector2`
- `fastf1_avg_sector3`

## Classificação dos outliers

### Outliers legítimos

Valores extremos explicáveis por eventos reais de corrida: Safety Car, VSC, falha mecânica
ou condições de pista úmida. Mantidos na base com `outlier_legitimo_flag = 1`.

### Outliers espúrios

Valores tecnicamente inválidos (ausentes ou menores/iguais a zero). Removidos da base.

### Outliers para revisão

Valores extremos plausíveis sem evidência suficiente para classificação automática imediata.
Mantidos com `outlier_revisao_flag = 1` para decisão posterior.

## Reconciliações pós-contexto (etapa 09)

Após a detecção inicial (etapa 06), foram aplicadas duas reconciliações em cascata na
etapa 09, quando informações contextuais adicionais já estavam disponíveis.

### Reconciliação 1 — Safety Car integrado

Após a integração definitiva de `safety_car_flag` via FastF1 `TrackStatus` (códigos 4, 6 e 7),
outliers classificados como `outlier_revisao` em corridas com `safety_car_flag = 1` foram
promovidos a `outlier_legitimo`.

A própria regra metodológica estabelece que extremos em corridas com Safety Car são eventos
reais de pista — o carro pode ter percorrido voltas lentas atrás do Safety Car sem que isso
represente erro de dado.

Resultado: **13 casos reclassificados**. Coluna `outlier_reclassificado_pos_contexto_flag`
marca os registros.

### Reconciliação 2 — Colunas não-feature

Outliers cujas colunas anômalas pertencem exclusivamente a variáveis FastF1 de tempo
(`fastf1_avg_sector1`, `fastf1_avg_sector2`, `fastf1_avg_sector3`, `fastf1_avg_lap_time`,
`fastf1_best_lap_time`) foram promovidos a `outlier_legitimo`.

Justificativa: essas colunas **não entram em X** no modelo final. O modelo prediz
`finish_position` a partir de coeficientes RAPM, forma recente e features de contexto —
não a partir de tempos de setor brutos. Um valor extremo em `fastf1_avg_sector1` não
invalida o resultado da corrida nem compromete a predição.

Resultado: **1 caso reclassificado** (ver caso específico abaixo). Coluna
`outlier_reclassificado_nao_feature_flag` marca o registro.

## Caso específico: Stroll — GP da Estíria 2021 (round 8)

| Campo | Valor |
|---|---|
| `season` | 2021 |
| `round` | 8 |
| `race_name` | Styrian Grand Prix |
| `driver_id` | stroll |
| `constructor_id` | aston_martin |
| `grid_position` | 9 |
| `finish_position` | 8 |
| `status` | +1 Lap |
| `fastf1_avg_sector1` | 20,04 s (outlier) |
| `safety_car_flag` | 0 |
| `corrida_chuva_flag` | 0 |

**Investigação:** Stroll completou a corrida em 8º lugar, +1 Lap, sem safety car e sem
chuva. O setor 1 elevado (20,04 s versus mediana do Red Bull Ring) é consistente com dano
mecânico leve ou percurso sob bandeira amarela local em algum momento da corrida — eventos
que produzem voltas lentas ocasionais sem acionar o SC/VSC global. O resultado final
(posição 8) é um dado válido de corrida.

**Decisão: `outlier_legitimo`** — via reconciliação por coluna não-feature (pass 2).

**Impacto no modelo:** nulo. `fastf1_avg_sector1` não entra em `X`. O dado relevante
para o RAPM (finish_position = 8) está correto.

## Estado final dos outliers (base 2018-2025)

| Classificação | Quantidade |
|---|---|
| `nao_outlier` | 2.917 |
| `outlier_legitimo` | 26 |
| `outlier_revisao` | **0** |

Todos os outliers detectados têm classificação definitiva. Nenhum caso permanece em revisão.

## Reprocessamento da normalização

Após a detecção, as colunas normalizadas foram recalculadas. Os parâmetros de normalização
foram ajustados com base na base 2018-2024 e aplicados também à base 2018-2025.

## Arquivos gerados

- `data/processed/historico_outliers_tratados_2018_2024.csv`
- `data/processed/historico_outliers_tratados_2018_2025.csv`
- `data/processed/outliers_removidos_2018_2024.csv`
- `data/processed/outliers_removidos_2018_2025.csv`
- `data/processed/outliers_revisao_2018_2025.csv` (vazio após reconciliações)
