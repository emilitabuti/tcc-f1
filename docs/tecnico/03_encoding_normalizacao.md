# 03 — Encoding e Normalização

## Contexto

Após o tratamento de DNFs, a base contém variáveis categóricas (circuito, construtor, composto de pneu, piloto) e numéricas (tempos de volta, posição de grid). Modelos de machine learning operam sobre números — as categóricas precisam ser convertidas, e as numéricas precisam estar em escalas comparáveis para o Ridge baseline.

O objetivo desta etapa é aplicar as transformações corretas para cada tipo de variável, sem introduzir informação do futuro e preservando a interpretabilidade dos dados.

---

## Fundamentação bibliográfica

| Decisão | Referência |
|---|---|
| OHE para circuito e construtor | Arquitetura, seção 2: "Sem ordem entre categorias" |
| Label Encoding ordinal para pneu | Arquitetura, seção 2: "Soft > Medium > Hard — ordem de desempenho" |
| Piloto como coeficiente RAPM (não OHE) | Arquitetura, seção 2: "Captura habilidade de forma contínua" — Henderson et al. [9] |
| XGBoost e RF dispensam normalização | Chen & Guestrin [19], Breiman [20] — algoritmos baseados em divisão de limiar (threshold splits) são invariantes a escala |
| Normalizar apenas para Ridge | Arquitetura, seção 2: "Z-score para variáveis numéricas contínuas; MinMaxScaler para GridPosition e Laps; XGBoost e Random Forest não exigem normalização — aplicar apenas para o baseline Ridge Regression" |

---

## Implementação

### Scripts envolvidos

| Script | O que faz | Saída |
|---|---|---|
| `src/encoding.py` | OHE em circuito e construtor, Label Encoding ordinal em pneu | `historico_encoded_*.csv`, encoders em `models/preprocessing/` |
| `src/normalizacao.py` | Z-score em contínuas, MinMax em grid_position e laps | `historico_normalizado_*.csv`, scalers em `models/preprocessing/` |

---

### Por que OHE para circuito e construtor?

Circuito e construtor são variáveis **nominais** — não existe ordem natural entre elas. Red Bull Ring não é "maior" que Monza. Mercedes não é "maior" que McLaren em alguma escala numérica. Usar Label Encoding (0, 1, 2, 3...) criaria uma ordem artificial que os algoritmos interpretariam como hierarquia, introduzindo viés.

O OHE cria uma coluna binária por categoria. O modelo aprende o efeito de cada circuito ou construtor de forma independente.

Implementação em `src/encoding.py`:

```python
colunas_categoricas = [coluna_circuito, "constructor_id"]
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoder.fit(df_2018_2024[colunas_categoricas])  # fit só em 2024
encoder.transform(df_2018_2025[colunas_categoricas])  # apply em 2025
```

O `handle_unknown="ignore"` é relevante para dados 2025+: se aparecer um circuito ou construtor novo (como acontece com o Cadillac em 2026), o encoder não quebra — simplesmente retorna zeros para aquela categoria.

**Detalhe sobre `circuit_id`:** o script mapeia `race_name` para `circuit_id` físico antes do OHE. Isso resolve o problema de um mesmo circuito ter nomes diferentes ao longo dos anos — ex: "Styrian Grand Prix" e "Austrian Grand Prix" ambos referem-se ao Red Bull Ring (`red_bull_ring`). Sem esse mapeamento, o encoder criaria duas colunas para o mesmo circuito. O mapa cobre 37 corridas distintas, incluindo os GPs extras de 2020 por COVID.

---

### Por que Label Encoding ordinal para composto de pneu?

Pneu é uma variável **ordinal**: existe uma relação técnica de dureza entre os compostos de pista seca. Soft desgasta mais rápido mas é mais aderente (mais rápido por volta). Hard dura mais mas é mais lento.

O mapeamento aplicado em `src/encoding.py`:

```python
COMPOUND_ORDINAL_MAP = {
    "HYPERSOFT": 6,   # mais suave, mais rápido — era 2018
    "ULTRASOFT": 5,   # era 2018
    "SUPERSOFT": 4,   # era 2018
    "SOFT":      3,   # composto padrão atual mais suave
    "MEDIUM":    2,
    "HARD":      1,   # mais duro, mais lento
    "INTERMEDIATE": 0,  # chuva — escala diferente
    "WET":          0,
    "UNKNOWN":      0,
}
```

Compostos de chuva (INTERMEDIATE, WET) e ausentes (UNKNOWN) recebem valor 0 porque não pertencem à mesma escala ordinal dos compostos de pista seca.

A feature final no modelo é `tire_compound_start` — o composto usado na volta de largada (primeiro stint). Valor mais alto = composto mais mole = estratégia de largada mais agressiva.

---

### Por que piloto não usa OHE nem Label?

Com ~50 pilotos distintos no período 2018-2025, OHE criaria 50 colunas binárias. Problemas:

1. **Alta dimensionalidade**: 50 colunas para uma variável, a maioria com pouquíssimos exemplos por piloto.
2. **Esparsidade**: pilotos com poucas corridas teriam coeficientes mal estimados ou zero.
3. **Sem continuidade temporal**: OHE não captura que um piloto melhorou ou piorou ao longo das temporadas — cada coluna binária é um efeito fixo estático.
4. **Irreproduzível em 2026**: novos pilotos em 2026 simplesmente não teriam coluna, tornando a predição impossível.

A solução adotada é o **coeficiente RAPM**: um número contínuo estimado para cada piloto e construtor via Ridge Regression sobre o histórico de corridas anteriores. Isso captura habilidade de forma contínua, temporal e extensível para pilotos novos (cold-start = 0.0). Documentado em detalhes no documento 05 (RAPM Ridge).

---

### Normalização: quais colunas, quais métodos

**Z-score** (`StandardScaler`, `src/normalizacao.py`, linhas 164-176):

Aplicado nas variáveis de telemetria FastF1 — contínuas com escalas muito diferentes entre si (tempo de volta em segundos, número de voltas, etc.):

```
laps, fastf1_laps_count, fastf1_avg_lap_time, fastf1_best_lap_time,
fastf1_avg_sector1, fastf1_avg_sector2, fastf1_avg_sector3,
fastf1_max_tyre_life, fastf1_stints_count, fastf1_pit_in_count, fastf1_pit_out_count
```

As colunas normalizadas recebem sufixo `_zscore` — as originais são preservadas.

**MinMax** (`MinMaxScaler`, `src/normalizacao.py`, linhas 179-190):

Aplicado em `grid_position` e `laps`:

```
grid_position → grid_position_minmax  (escala 1–22)
laps          → laps_minmax           (escala variável por corrida)
```

---

### Por que XGBoost e RF não precisam de normalização?

Árvores de decisão fazem divisões binárias em limiares: `grid_position > 10?`. O valor exato da escala não importa — o que importa é a ordenação dos valores. Aplicar z-score ou MinMax em `grid_position` não muda a ordenação, portanto não muda as divisões e não muda as predições.

Normalizar para árvores seria trabalho sem efeito. A arquitetura é explícita: "XGBoost e Random Forest não exigem normalização".

O Ridge Regression, por outro lado, minimiza uma função de custo quadrática com penalização L2. Features em escalas diferentes causam penalização desequilibrada: uma feature em milissegundos (ex: `fastf1_avg_lap_time` ≈ 90.000) receberia coeficiente minúsculo enquanto uma em posições inteiras (ex: `grid_position` ≈ 10) receberia coeficiente maior — não por importância real, mas por diferença de escala.

---

### O z-score calculado aqui introduz leakage no walk-forward?

**Resposta: não — mas há um risco latente que deve ser monitorado.**

O scaler é ajustado (`fit`) na base 2018-2024 e apenas aplicado (`transform`) na base 2018-2025. Isso significa que os parâmetros de normalização (média e desvio padrão) não veem dados de 2025.

Porém, na validação walk-forward com três folds (2023, 2024, 2025), o scaler foi ajustado em toda a base 2018-2024 — não apenas no subconjunto de treino de cada fold. No fold 1 (treino 2018-2022, validação 2023), o scaler já conhece a distribuição de 2023 e 2024.

**Isso é um problema?** Para os modelos de árvore (LightGBM, RF, XGBoost): não, porque as colunas `_zscore` e `_minmax` não entram no X final. As 15 features do modelo são `qualifying_position`, `recent_form_5`, etc. — nenhuma delas é coluna `_zscore`.

Para o Ridge baseline, que usa as mesmas 15 features diretamente (sem colunas `_zscore`), o StandardScaler é reaplicado dentro do walk-forward por fold. O risco latente existe se alguém modificar o pipeline para usar as colunas `_zscore` como features diretas sem refazer o fit por fold.

---

### Serialização dos scalers e encoders

Os artefatos são salvos em `models/preprocessing/` com `joblib`:

| Artefato | Propósito |
|---|---|
| `onehot_encoder_historico_fastf1.joblib` | Reproduzir o OHE exato nos dados 2026 |
| `schema_encoding_historico_fastf1.json` | Validar que as categorias e colunas batem |
| `standard_scaler_historico.joblib` | Reproduzir z-score com mesma média/DP |
| `minmax_scaler_historico.joblib` | Reproduzir MinMax com mesmo mín/máx |

A serialização é necessária porque os parâmetros (médias, desvios, categorias do OHE) foram estimados no dataset 2018-2024. Para processar dados de 2026 no mesmo espaço de features, os scalers precisam ser reaplicados com os mesmos parâmetros — não re-estimados nos dados novos.

---

## Resultados obtidos

Do `relatorio_03_encoding.txt`:

| Base | Dimensão entrada | Colunas circuito OHE | Colunas construtor OHE | Dimensão saída |
|---|---|---|---|---|
| 2018-2024 (enriquecida FastF1) | (2.524, ~30) | 29 | 16 | (2.524, ~75) |
| 2018-2025 (enriquecida FastF1) | (2.943, ~30) | 29 | 16 | (2.943, ~75) |

29 circuitos únicos × 16 construtores únicos no período 2018-2024.

Do `relatorio_04_normalizacao.txt`: 11 colunas com z-score, 2 colunas com MinMax. Nulos nessas colunas preenchidos com mediana da base 2018-2024 antes do fit.

---

## Avaliação crítica

**Pontos fortes:**
- OHE ajustado em 2018-2024 e aplicado em 2025 com `handle_unknown="ignore"` — extensível para novas categorias.
- Artefatos serializados garantem reprodutibilidade exata.
- Colunas originais preservadas — normalização não sobrescreve dados.

**Limitações:**
- O mapeamento `race_name → circuit_id` é hardcoded: 37 corridas mapeadas. Qualquer GP novo (ex: Madrid 2026, que aparece nos raw data) precisa ser adicionado manualmente ou o pipeline quebra com `ValueError`.
- Os compostos HYPERSOFT, ULTRASOFT e SUPERSOFT são de 2018 e foram descontinuados. Em temporadas recentes (2023+) apenas SOFT, MEDIUM e HARD são usados — o `compound_ordinal` efetivamente varia entre 1 e 3 na base majoritária.
- O scaler de normalização foi ajustado em toda a base 2018-2024, não por fold do walk-forward. Para as 15 features atuais isso é inócuo, mas é uma fragilidade se o pipeline for modificado.

---

## Convergência com a literatura

| Aspecto | Alinhado | Divergente | Observação |
|---|---|---|---|
| OHE para categorias nominais | ✅ | — | Padrão da literatura e da arquitetura |
| Ordinal para pneu (performance) | ✅ | — | Arquitetura seção 2: "Soft > Medium > Hard" |
| Piloto via RAPM, não OHE | ✅ | — | Henderson et al. [9], arquitetura seção 2 |
| Normalizar só para Ridge | ✅ | — | Arquitetura explícita; coerente com [19][20] |
| Scaler ajustado no treino, não no teste | ✅ | — | Sem leakage para as 15 features finais |
