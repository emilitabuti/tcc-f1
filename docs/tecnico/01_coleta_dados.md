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

A integração é feita em `src/limpeza_ergast_fastf1.py`. O join usa `driver_id + season + round` como chave composta. O FastF1 usa códigos de três letras (ex: `HAM`) que precisam ser mapeados para os IDs do Ergast (ex: `hamilton`). O mapeamento está hardcoded no dicionário `DRIVER_CODE_TO_ID` (44 entradas) no início do script.

O FastF1 é agregado por `RaceID` antes do join — cada piloto tem dezenas de voltas no FastF1, que são reduzidas a métricas por corrida (média de tempo de volta, composto predominante, contagem de pit stops, etc.).

Resultado do merge (do `relatorio_01`):
- 3.458 linhas no Ergast após concatenação 2018-2025
- 3.452 linhas no FastF1 após agregação por RaceID
- **6 registros sem correspondência FastF1** — mantidos na base com valores NaN nas colunas FastF1

### O que acontece quando um GP está no Ergast mas não no FastF1?

6 registros ficaram sem correspondência (confirmado no `relatorio_01`). O join é `how="left"` mantendo todos os registros do Ergast. Registros sem FastF1 recebem `NaN` nas colunas de telemetria e são tratados na etapa de valores ausentes (Etapa 05).

Causa provável: sessões com problemas de extração ou GPs sem dados de telemetria disponíveis no FastF1 para aquela combinação season/round.

### Schema do OpenF1 vs. Ergast/FastF1

| Dimensão | Ergast/FastF1 | OpenF1 |
|---|---|---|
| Identificador de piloto | `driver_id` (string, ex: `hamilton`) | `driver_number` (inteiro, ex: 44) |
| Identificador de corrida | `season + round` | `meeting_key` (inteiro único) |
| Cobertura histórica | 2018–2024 (Ergast) | 2023+ (cobertura crescente) |
| Latência | Dados finais consolidados | Dados disponíveis durante/após a sessão |

A diferença de identificadores exige um mapeamento `driver_number → driver_id` para qualquer join com o pipeline histórico. Este mapeamento foi implementado em `src/mapear_openf1_ergast.py` e documentado em `docs/mapeamento_openf1_ergast.md`.

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

2020 tem 340 registros (e não 440) por conta do calendário reduzido pela pandemia de COVID-19 — 17 corridas em vez de 20-22.

---

## Avaliação crítica

**Pontos fortes:**
- Três fontes complementares garantem cobertura completa das variáveis necessárias.
- Sem nulos nas colunas essenciais após a limpeza — base de partida limpa.
- RaceID com lógica simples, verificável e sem colisões.
- Mapeamento FastF1 → Ergast hardcoded é auditável e completo para o período coberto.

**Limitações:**
- Corte em 2018 (não 2014) reduz o período de treino em relação aos benchmarks da literatura. Comparações diretas de MAE com Henderson et al. [9] devem mencionar essa diferença.
- Os 6 registros sem FastF1 entram com NaN em todas as features de telemetria. Se forem corridas relevantes (ex: GPs com pilotos de alto desempenho), isso pode introduzir viés pontual.
- O mapeamento `driver_number → driver_id` é hardcoded e precisa ser atualizado manualmente quando novos pilotos entram no grid (evidenciado pela necessidade de `DRIVER_NUMBER_TO_ID_2026_NEW` no `update_openf1_2026.py`).
- A Ergast API original foi descontinuada e migrada para Jolpica. Scripts que referenciam diretamente `ergast.com` precisariam ser atualizados.

**Riscos:**
- 45 registros com `grid_position = 0` são mantidos com flag. Se entrarem no modelo sem tratamento, podem introduzir ruído na feature `qualifying_position` / `grid_position`.

---

## Convergência com a literatura

| Aspecto | Alinhado | Divergente | Observação |
|---|---|---|---|
| Usar Ergast como fonte de resultados históricos | ✅ | — | Padrão da literatura [2][3][9] |
| FastF1 para qualifying e telemetria | ✅ | — | Seguido por [2][7] |
| Corte em 2014 (era híbrida) | — | ⚠️ | Implementado como 2018 por limitação do FastF1 |
| Chave piloto-corrida única | ✅ | — | Equivalente ao `RaceID` implícito em [9] |
| Join Ergast + FastF1 por piloto + round | ✅ | — | Mesma estratégia de [2] |
