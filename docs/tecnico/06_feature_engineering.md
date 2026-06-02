# 06 — Feature Engineering

## Contexto

Com a base limpa e os coeficientes RAPM calculados, o objetivo desta etapa é construir as features que entrarão no modelo. Todas as features são computadas **causalmente**: para a corrida r, cada feature usa somente informação disponível antes de r acontecer. Nenhuma feature pode depender do resultado da corrida que está sendo predita.

Esta etapa se divide em dois scripts:
- `src/09_preparar_base_feature_engineering.py` — features de circuito, clima histórico, pit stops e `track_complexity` enriquecida.
- `src/feature_engineering_parte_1.py` — features históricas de piloto e construtor.

---

## Fundamentação bibliográfica

| Feature | Referência principal |
|---|---|
| `recent_form_5`, `recent_form_3` | Ruan et al. [2] — RF+SHAP paper: Recent Form Indicator |
| `driver_coef_rapm`, `constructor_coef_rapm` | Henderson et al. [9] — RAPM paper |
| `driver_dnf_rate`, `constructor_dnf_rate` | Ruan et al. [2] — DNF Score |
| `driver_constructor_synergy` | Ruan et al. [2] — Driver-Constructor Synergy |
| `track_complexity` | Ruan et al. [2], Heilmeier et al. [6] |
| `weather_impact_factor` | Ruan et al. [2] — Weather Impact Factor (adaptado para histórico causal) |
| `avg_pit_stops_circuit` | Heilmeier et al. [6] |
| `qualifying_position` | Barra et al. [3], Koopman [5] — feature dominante na literatura |
| `grid_penalty` | Adição própria — sem citação específica na literatura revisada |
| `season_factor` | Razoável conceitualmente — sem citação direta |

---

## Tabela de features — fórmula, causalidade e cold-start

### Grupo Grid

| Feature | Fórmula | Mecanismo causal | Cold-start |
|---|---|---|---|
| `qualifying_position` | Posição obtida na sessão de qualifying (Q1/Q2/Q3) | Acontece antes da corrida | `grid_position` como fallback (~0,6% dos casos) |
| `grid_penalty` | `grid_position - qualifying_position` | Calculado antes da largada | 0 quando penalidade desconhecida |

`qualifying_position` não estava na arquitetura original — que previa apenas `grid_position`. Foi adicionada porque o qualifying acontece no sábado, antes da corrida de domingo, e representa o desempenho *puro* do piloto no carro sem tráfego. A correlação com o target (r=0.772) é a maior de todas as features — superior à `grid_position` (r=0.753). A mudança foi documentada e justificada.

`grid_penalty` captura penalidades de grid (motores extras, comportamento na corrida anterior). Pilotos com penalidade saem mais atrás do que qualificaram. A feature mede essa distância.

---

### Grupo Forma Recente

| Feature | Fórmula | Mecanismo causal | Cold-start |
|---|---|---|---|
| `recent_form_5` | Média ponderada das últimas 5 `finish_position` (pesos: corrida mais recente=5, segunda mais recente=4, …, quinta=1) | `historico.append(resultado)` *após* calcular a feature — `historico` nunca inclui a corrida atual | 0.0 (sem histórico) |
| `recent_form_3` | Idem com últimas 3 corridas (pesos: 3, 2, 1) | Idem | 0.0 |

Implementação em `feature_engineering_parte_1.py` (função `adicionar_recent_form`):

```python
for idx, row in grupo.iterrows():
    df.loc[idx, "recent_form_5"] = weighted_recent_form(historico, n_corridas=5)
    df.loc[idx, "recent_form_3"] = weighted_recent_form(historico, n_corridas=3)
    historico.append(row["finish_position"])  # adiciona APÓS calcular
```

Maior `recent_form` = posições recentes piores (posição 1 é melhor, mas vale 1; posição 10 vale 10). Portanto correlação positiva com `finish_position` é esperada (r=0.710 para `recent_form_5`).

**Sobre `recent_form_3`:** foi criada mas depois removida na seleção de features por multicolinearidade severa com `recent_form_5` (r=0.987). Documentada aqui para rastreabilidade — o RFE excluiu `recent_form_3` do conjunto final.

---

### Grupo Piloto

| Feature | Fórmula | Mecanismo causal | Cold-start |
|---|---|---|---|
| `driver_coef_rapm` | Coeficiente Ridge causal por corrida (ver documento 05) | Treina só em corridas anteriores a r | 0.0 |
| `driver_experience` | `cumcount()` — total de corridas anteriores do piloto | `cumcount()` por piloto ordenado por `race_order` | 0 na primeira corrida |
| `driver_wins_total` | `cumsum().shift(1)` de `driver_win_flag` | `shift(1)` garante que a vitória da corrida atual não entra | 0 na primeira corrida |
| `driver_dnf_rate` | `driver_dnf_before / driver_starts_before` — calculado no `historico_dnf_classificado` com `shift(1)` | Calculado no dataset de DNFs classificados, mesclado depois na base principal | 0.0 sem histórico |

Implementação de `driver_experience` e `driver_wins_total` em `feature_engineering_parte_1.py`:

```python
df["driver_experience"] = df.groupby("driver_id").cumcount()
df["driver_wins_total"] = (
    df.groupby("driver_id")["driver_win_flag"]
    .transform(lambda s: s.cumsum().shift(1).fillna(0))
)
```

`driver_dnf_rate` é calculada no histórico DNF classificado (que contém *todos* os registros, incluindo DNFs) e depois mesclada na base principal (que contém apenas classificados). Isso é necessário porque a taxa de DNF histórica deve incluir as corridas que o piloto abandonou — excluí-las do cálculo produziria uma taxa artificialmente baixa (zero em casos extremos).

---

### Grupo Construtor

| Feature | Fórmula | Mecanismo causal | Cold-start |
|---|---|---|---|
| `constructor_coef_rapm` | Coeficiente Ridge causal (ver documento 05) | Idem piloto | 0.0 |
| `constructor_wins_total` | `cumsum().shift(1)` de vitórias do construtor por corrida (agrega pilotos) | `shift(1)` por grupo `constructor_id + race_order` | 0 na primeira corrida |
| `constructor_dnf_rate` | `constructor_dnf_car_before / constructor_entries_before` (só falhas mecânicas, não acidentes) | `shift(1)` no histórico DNF por construtor | 0.0 sem histórico |

A `constructor_dnf_rate` usa somente `dnf_car_flag` (falhas mecânicas), não `dnf_driver_flag` (acidentes). A justificativa: a confiabilidade mecânica do carro é um atributo do construtor. Acidentes do piloto não devem contaminar a avaliação de confiabilidade da equipe.

---

### Sinergia Piloto-Construtor

| Feature | Fórmula | Mecanismo causal | Cold-start |
|---|---|---|---|
| `driver_constructor_synergy` | `expanding().mean().shift(1)` de `-finish_position` por par `(driver_id, constructor_id)` | `shift(1)` dentro do par | 0.0 na primeira corrida do par |

```python
df["performance_score"] = -df["finish_position"]
df["driver_constructor_synergy"] = (
    df.groupby(["driver_id", "constructor_id"])["performance_score"]
    .transform(lambda s: s.expanding().mean().shift(1))
)
```

Usa `-finish_position` (não `finish_position`), então valores mais altos indicam melhor sinergia histórica. Captura algo que `recent_form_5` não captura: o histórico específico do piloto *com aquela equipe*. Um piloto que performou bem em equipes anteriores mas está em dificuldades na equipe atual terá `recent_form_5` em degradação mas `driver_constructor_synergy` com novo par = 0.0 (cold-start), refletindo a incerteza real.

A correlação de r=-0.87 com `recent_form_5` é alta mas não eliminatória — ambas aparecem no top-5 de todos os modelos de árvore (documentado no documento 10).

---

### Grupo Circuito

| Feature | Fórmula | Mecanismo causal | Cold-start |
|---|---|---|---|
| `track_complexity` | `0.35×corners_norm + 0.25×length_km_norm + 0.20×altitude_norm + 0.10×circuit_type + 0.10×incident_rate_hist_norm` | Componentes estáticos: dados do circuito existentes antes da corrida. Componente `incident_rate_hist_norm`: taxa histórica causal (ver abaixo) | Média global 2018-2024 para `incident_rate_hist_norm` |
| `incident_rate_hist_norm` | Taxa histórica de SC/VSC no circuito: `expanding().mean().shift(1)` por `circuit_id` | `shift(1)` por circuito | Taxa global 2018-2024 |
| `altitude_m` | Altitude do circuito em metros (dado estático de `circuitos_manual.csv`) | Estático — existe antes da corrida | N/A (estático) |

`track_complexity` tem dois subcomponentes com naturezas diferentes:
- **Estáticos** (corners, length, altitude, circuit_type): propriedades físicas permanentes do circuito.
- **Causal histórico** (`incident_rate_hist_norm`): proporção de corridas anteriores naquele circuito que tiveram safety car. Calculada com `expanding().mean().shift(1)`.

Os pesos (0.35, 0.25, 0.20, 0.10, 0.10) são **arbitrários** — sem calibração empírica. Este é um ponto de fragilidade documentável. A correlação final com o target é próxima de zero (r=-0.012), o que pode indicar que o índice composto não captura o que importa para posição final, ou que o efeito existe mas é mediado por outras features.

---

### Grupo Estratégia e Pneu

| Feature | Fórmula | Mecanismo causal | Cold-start |
|---|---|---|---|
| `tire_compound_start` | `compound_ordinal` da largada (Soft=3, Medium=2, Hard=1, Wet=0) — ver documento 03 | Composto de largada é decidido antes da corrida | MEDIUM (2) como fallback |
| `avg_pit_stops_circuit` | `expanding().mean().shift(1)` de `fastf1_pit_in_count` médio por corrida no circuito | `shift(1)` por `circuit_id` | Média global anterior como fallback |

`avg_pit_stops_circuit` foi recalculada causalmente na etapa 09 — substituiu uma versão estática que usava a média de todo o período. A versão estática foi preservada em `avg_pit_stops_circuit_static_global` para auditoria. Linhas com cold-start nessa feature: 511 de 2.943 (17,4%) — correspondentes às primeiras corridas de cada circuito na base.

---

### Grupo Clima

| Feature | Fórmula | Mecanismo causal | Cold-start |
|---|---|---|---|
| `weather_impact_factor` | `expanding().mean().shift(1)` de `weather_impact_observed` por `circuit_id` | Usa histórico de clima nas corridas *anteriores* do mesmo circuito | 0.0 na primeira corrida do circuito |

`weather_impact_observed` por corrida é calculado como:

```
(humidity/100 + 2×rain_binary + (1 - air_temp/45)) / 4
```

Esse valor *observado* (que usa dados reais da corrida em si) fica **fora de X** — é apenas histórico. A feature `weather_impact_factor` que entra no modelo é a média desse índice nas corridas *anteriores* do mesmo circuito. Isso resolve o leakage original: ao predizer uma corrida, o modelo usa o padrão histórico de clima daquele circuito, não o clima real da corrida atual.

A RFE posterior excluiu `weather_impact_factor` do conjunto final de 15 features — o sinal histórico de clima foi insuficiente para adicionar valor marginal ao modelo.

---

### Grupo Temporal

| Feature | Fórmula | Mecanismo causal | Cold-start |
|---|---|---|---|
| `season_factor` | `int(season)` — ano da corrida | Trivialmente causal | N/A |

`season_factor` captura a evolução tecnológica ao longo das temporadas — carros de 2025 são diferentes dos de 2018. A hipótese é que temporadas mais recentes revelam padrões diferentes de desempenho. Correlação com o target próxima de zero (r=0.035), mas o RFE manteve a feature no conjunto final (rank 6 por gain, gain=56).

---

## Resultados obtidos

Do `relatorio_11_feature_engineering_parte_1.txt`:

| Validação | Resultado |
|---|---|
| Linhas entrada | 2.524 (versão 2018-2024) |
| Linhas saída | 2.524 |
| RaceID duplicados | 0 |
| NaN em qualquer feature | 0 |
| `driver_dnf_rate` nonzero | 2.096 de 2.524 (83%) |
| `constructor_dnf_rate` nonzero | 2.372 de 2.524 (94%) |
| `driver_dnf_rate` min/max | 0.0 / 1.0 |
| `constructor_dnf_rate` min/max | 0.0 / 1.0 |

Do `relatorio_09_preparacao_feature_engineering.txt`:

| Validação | Resultado |
|---|---|
| Linhas base FE-ready 2018-2025 | 2.943 |
| Colunas | 124 |
| Erros bloqueantes | 0 |
| Corridas com Safety Car 2018-2025 | 126 de 173 (73%) |
| Linhas cold-start `avg_pit_stops_circuit` | 511 (17,4%) |
| Outliers em revisão final | 0 (todos reclassificados) |

---

## Avaliação crítica

**Por que `recent_form_3` foi criada e depois removida?**

A arquitetura previa ambas as features para "capturam janelas diferentes". Na prática, r=0.987 entre as duas torna a diferença de janela empiricamente irrelevante nesse dataset. O RFE confirmou: remover `recent_form_3` não piora o MAE. A decisão de manter apenas `recent_form_5` está alinhada com a janela maior citada pelo RF+SHAP paper [2].

**`driver_wins_total` foi removida pelo RFE (rank 16) — por quê?**

A arquitetura a previa como feature. O RFE a excluiu porque seu sinal está majoritariamente capturado em `driver_coef_rapm` — ambas medem o histórico de sucesso do piloto, mas o RAPM o faz de forma mais contínua e temporal. Manter as duas seria redundância parcial.

**Pesos de `track_complexity` são arbitrários:**

A literatura (Ruan et al. [2]) cita complexidade de circuito como feature relevante mas não especifica pesos para componentes individuais. Os valores 0.35/0.25/0.20/0.10/0.10 foram definidos sem calibração empírica. A correlação final com o target (r=-0.012) é próxima de zero — possíveis explicações: (1) a complexidade do circuito afeta o *número* de acidentes mas não quem *vence* (determinado pelo carro/piloto); (2) o efeito é capturado indiretamente pelos coeficientes RAPM históricos do circuito; (3) os pesos estão mal calibrados.

**`season_factor` como feature discutível:**

Um modelo walk-forward com treino até 2024 e validação em 2025 verá `season_factor=2025` como valor fora do range de treino. Modelos de árvore não extrapolam — `season_factor` ficará na divisão mais próxima do treino (≤2024). O efeito real provavelmente é capturado via time-decay no walk-forward, tornando `season_factor` parcialmente redundante. O RFE manteve a feature (rank 6), então há algum sinal residual não capturado pelo time-decay.

---

## Convergência com a literatura

| Feature | Alinhado | Divergente | Observação |
|---|---|---|---|
| `recent_form_5` pesos (5,4,3,2,1) | ✅ | — | Ruan et al. [2] |
| `driver_dnf_rate` via histórico piloto | ✅ | — | Ruan et al. [2] |
| `constructor_dnf_rate` só falhas mecânicas | ✅ | — | Separação piloto/carro da arquitetura |
| `driver_constructor_synergy` via desempenho histórico | ✅ | — | Ruan et al. [2] |
| `qualifying_position` em vez de `grid_position` | ✅ | — | Barra et al. [3]: correlação esperada r≈0.71; observada r=0.77 |
| Pesos `track_complexity` calibrados empiricamente | — | ⚠️ | Pesos arbitrários — sem calibração |
| `weather_impact_factor` como histórico causal | ✅ | — | Correção do leakage original; RFE excluiu a feature do conjunto final |
| `grid_penalty` com referência bibliográfica | — | ⚠️ | Feature adicionada sem citação específica |
| `season_factor` com referência bibliográfica | — | ⚠️ | Razoável conceitualmente, sem citação direta |
