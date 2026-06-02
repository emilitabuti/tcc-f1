# 04 — Valores Ausentes e Outliers

## Contexto

Com a base encoded e normalizada, as etapas 05 e 06 resolvem dois problemas distintos: lacunas nos dados (valores ausentes) e valores extremos (outliers). São problemas diferentes com estratégias diferentes — imputar um outlier ou remover um valor ausente são decisões que afetam a integridade do sinal preditivo.

---

## Fundamentação bibliográfica

| Decisão | Referência |
|---|---|
| KNN para qualifying ausente | Arquitetura, seção 2: "Qualifying (0,07% ausente) → Imputação KNN" |
| Mediana por circuito para tempos de volta | Arquitetura, seção 2: "Numéricas contínuas (tempos de volta) → Mediana do circuito naquele ano" |
| Moda por corrida para composto de pneu | Arquitetura, seção 2: "Categóricas (composto de pneu) → Moda da corrida" |
| Critério 3σ por circuito | Arquitetura, seção 2: "Critério: valores acima de 3 desvios padrão da média por circuito — mesmo critério do Advanced ML paper" — Barra et al. [3] |
| Outliers legítimos mantidos com flag | Arquitetura, seção 2: "Outliers legítimos como safety car e falhas mecânicas: manter com flag binária `safety_car_flag = 1`" |

---

## Implementação — Etapa 05: Valores Ausentes

**Script:** `src/tratamento_valores_ausentes.py`

### Estratégia por tipo de variável

| Tipo | Colunas | Estratégia | Lógica |
|---|---|---|---|
| Tempos de volta e setores | `fastf1_avg_lap_time`, `fastf1_best_lap_time`, `fastf1_avg_sector1/2/3` | Mediana do circuito naquele ano | Tempos variam muito entre circuitos — a mediana global distorceria Monaco (lento) com Monza (rápido) |
| Composto de pneu | `compound_normalizado` | Moda da corrida | Na mesma corrida, a maioria dos pilotos usa o mesmo composto de largada |
| Qualifying | Colunas com "qualifying", "q1", "q2", "q3" | KNN com 5 vizinhos | Preserva a estrutura local: pilotos com desempenho similar têm posição de qualifying similar |

**Fallbacks para tempos de volta (na ordem):**
1. Mediana do circuito naquele ano — caso geral
2. Mediana do ano inteiro — se aquele circuito não tem observações suficientes
3. Mediana global — se nem o ano tem dados
4. 0 — caso extremo sem nenhuma referência

### Por que mediana e não média para tempos?

A média é sensível a outliers. Um tempo de volta absurdamente alto (volta atrás do safety car, warm-up lap) puxaria a média para cima e a imputação resultaria num tempo irreal. A mediana é robusta a esses extremos.

### Por que KNN para qualifying e não mediana?

A mediana da posição de qualifying de um circuito seria, em geral, ~10 (posição central do grid). Isso não diz nada sobre o piloto específico. O KNN usa 5 vizinhos com features similares (`season`, `round`, `grid_position`, `compound_ordinal`, `fastf1_avg_lap_time`, `fastf1_best_lap_time`) para inferir uma posição de qualifying plausível para aquele piloto naquele contexto. Um piloto rápido que perdeu o qualifying terá vizinhos rápidos, gerando uma imputação mais próxima da realidade.

### Resultado da imputação (do `relatorio_05`):

**Todos os campos com 0 nulos antes e depois da imputação.**

Nenhuma coluna de tempo, composto ou qualifying apresentou valores ausentes na base processada até essa etapa. O relatório confirma: `'nulos_antes': 0, 'nulos_depois': 0` para todas as colunas de tempo de volta.

**Nota importante sobre o KNN para qualifying:** o relatório registra `{'colunas_qualifying': [], 'aplicado': False}`. A imputação KNN de qualifying **não foi aplicada nesta etapa** porque as colunas de qualifying (`qualifying_position`, `Q1`, `Q2`, `Q3`) ainda não estavam na base neste ponto do pipeline — elas são integradas via FastF1 nas etapas 07 e 09. A arquitetura menciona KNN como estratégia para qualifying ausente, mas sua aplicação efetiva ocorre no momento da integração do FastF1 qualifying, não nesta etapa isolada.

---

## Implementação — Etapa 06: Outliers

**Script:** `src/tratamento_outliers.py`

### Critério de detecção

Outlier é qualquer valor acima de **3 desvios padrão da média do circuito** nas seguintes colunas de telemetria FastF1:

```
fastf1_avg_lap_time, fastf1_best_lap_time,
fastf1_avg_sector1, fastf1_avg_sector2, fastf1_avg_sector3
```

O critério é **por circuito**, não global. A mesma lógica de Barra et al. [3] (Advanced ML paper). A razão: um tempo de 90s é normal em Monza mas absurdo em Monaco (onde tempos típicos são ~75s). Aplicar um único limiar global misturaria circuitos lentos e rápidos, classificando erroneamente pilotos em circuitos lentos como outliers.

Implementação (`tratamento_outliers.py`, linha 244–254):

```python
media_circuito = df.groupby("circuito_derivado")[coluna].transform("mean")
desvio_circuito = df.groupby("circuito_derivado")[coluna].transform("std")
limite_superior = media_circuito + (3 * desvio_circuito)
outlier_coluna = df[coluna] > limite_superior
```

Apenas limite **superior** — tempos de volta não podem ser negativos, então um outlier inferior seria fisicamente impossível e já seria capturado como `valor ≤ 0`.

### Classificação dos outliers detectados

Após marcar o `outlier_flag = 1`, cada outlier é classificado em três categorias:

| Tipo | Critério | Ação |
|---|---|---|
| `outlier_legitimo` | `safety_car_flag = 1` **ou** `dnf_car_flag = 1` **ou** status indica falha mecânica **ou** corrida teve pneu WET/INTERMEDIATE | **Mantido** com flag — evento real |
| `outlier_revisao` | Outlier detectado mas sem evidência de causa real | **Mantido** com flag — não removido automaticamente |
| `outlier_espurio` | Valor tecnicamente inválido: NaN ou ≤ 0 | **Removido** |

A prioridade de classificação segue esta ordem no código: primeiro verifica `cond_espurio`, depois `cond_legitimo`. Um outlier só é espúrio se for tecnicamente inválido (NaN ou zero negativo) — não por ser extremo.

### Sobre o `safety_car_flag` nesta etapa

Existe uma nuance importante: quando essa etapa executa, a coluna `safety_car_flag` **não existe ainda** na base. O script cria o placeholder com zeros (linha 164):

```python
if "safety_car_flag" not in df.columns:
    df["safety_car_flag"] = 0
```

A flag real de safety car — derivada do `FastF1 TrackStatus 4/6/7` — é integrada na etapa 09 (`09_preparar_base_feature_engineering.py`). Isso significa que na classificação de outliers desta etapa, **nenhum outlier foi classificado como legítimo por safety car** — apenas por `dnf_car_flag` ou `corrida_chuva_flag`.

Essa mesma `safety_car_flag` que foi criada como feature real na etapa 09 é o que foi depois identificado como **leakage**: ela registra se aquela corrida específica teve safety car, informação que só existe após a corrida ocorrer. Seu papel aqui (proteger outliers de tempo de volta lento) era legítimo — o problema foi tê-la incluído como feature de entrada no modelo preditivo. A substituição por `incident_rate_hist_norm` (taxa histórica causal) resolve o leakage sem perder o sinal de contexto de segurança.

---

## Resultados obtidos

### Etapa 05 — Valores Ausentes

| Base | Nulos antes | Nulos depois | KNN aplicado |
|---|---|---|---|
| 2018-2024 | 0 (todas as colunas) | 0 | Não — qualifying não disponível nesta etapa |
| 2018-2025 | 0 (todas as colunas) | 0 | Não |

### Etapa 06 — Outliers (do `relatorio_06`)

**Base 2018-2024:**

| Tipo | Quantidade |
|---|---|
| `nao_outlier` | 2.510 |
| `outlier_revisao` | 10 |
| `outlier_legitimo` | 4 |
| `outlier_espurio` | 0 |
| **Total** | **2.524** |
| **Removidos** | **0** |

**Base 2018-2025:**

| Tipo | Quantidade |
|---|---|
| `nao_outlier` | 2.917 |
| `outlier_revisao` | 14 |
| `outlier_legitimo` | 12 |
| `outlier_espurio` | 0 |
| **Total** | **2.943** |
| **Removidos** | **0** |

**Nenhum registro foi removido.** Todos os outliers detectados foram classificados como legítimos ou para revisão — nenhum foi tecnicamente inválido (NaN ou ≤ 0).

---

## Avaliação crítica

**Por que 0 outliers espúrios é plausível?**

A base já passou por dois filtros anteriores: (1) a Etapa 01 removeu registros com `finish_position` ou `grid_position` nulos, e (2) a Etapa 02 removeu DNFs. Quem chegou até aqui completou a corrida e tem dados mínimos válidos. A ausência de outliers espúrios (NaN ou ≤ 0 em tempos de volta) é consistente com essa filtragem prévia.

**Os 14 registros em `outlier_revisao` (2018-2025) são um risco?**

Esses registros têm tempos de volta extremos mas não foi possível identificar automaticamente a causa (sem safety car flag real, sem falha mecânica, sem chuva). Eles permanecem na base com `outlier_revisao_flag = 1`. Para a modelagem com árvores, esses 14 registros em 2.943 (~0,5%) têm impacto negligenciável — os modelos de árvore são robustos a outliers. Para o Ridge baseline, podem influenciar levemente os coeficientes.

**Limitação da etapa 06 — `safety_car_flag` como placeholder:**

A classificação de outliers legítimos via safety car ficou comprometida porque a flag não existia ainda. Os 4 legítimos de 2018-2024 e 12 de 2018-2025 foram classificados por `dnf_car_flag` ou `corrida_chuva_flag` — que são causas igualmente válidas para tempo de volta alto (falha mecânica lenta, corrida na chuva). Mas casos de outlier causados *só* por safety car, sem chuva e sem falha mecânica, entraram em `outlier_revisao`.

**KNN para qualifying: posição no pipeline a ser documentada no TCC**

A arquitetura descreve KNN para qualifying como estratégia desta etapa. Na implementação real, o KNN não foi aplicado aqui porque o qualifying chega mais tarde. O TCC deve documentar claramente que o tratamento de ausentes de qualifying ocorre via integração direta do FastF1 qualifying na etapa 07/09, não via imputação KNN desta etapa.

---

## Convergência com a literatura

| Aspecto | Alinhado | Divergente | Observação |
|---|---|---|---|
| Critério 3σ por circuito | ✅ | — | Mesma estratégia de Barra et al. [3] |
| Mediana do circuito para tempos | ✅ | — | Arquitetura seção 2 |
| Moda da corrida para pneu | ✅ | — | Arquitetura seção 2 |
| KNN para qualifying | ⚠️ | — | Previsto na arquitetura; não aplicado nesta etapa por ausência do dado — qualifying chega nas etapas 07/09 |
| Outliers legítimos mantidos com flag | ✅ | — | Arquitetura seção 2; coerente com decisão de manter eventos reais de corrida |
| `safety_car_flag` como placeholder aqui | ⚠️ | — | Flag criada com zeros nesta etapa; valor real integrado na etapa 09 e depois identificado como leakage no modelo |
