# 02 — Limpeza e Tratamento de DNF

## Contexto

Após a coleta, a base bruta contém registros de todo piloto que participou de cada corrida — incluindo aqueles que abandonaram antes do fim. O problema é que a posição final de um piloto que abandonou não é comparável à de um que completou a corrida: um piloto que largou em 3º e abandona por falha mecânica na volta 10 é registrado em último lugar, mas seu desempenho real foi completamente diferente disso.

O objetivo desta etapa é: (1) remover registros inválidos da base bruta, e (2) decidir o que fazer com os registros de abandono.

---

## Fundamentação bibliográfica

| Decisão | Referência |
|---|---|
| Variante DNF Excluded como abordagem padrão | Henderson et al. [9] — RAPM paper (MAE benchmark de 2.3 adotado com DNF Excluded) |
| Classificação de DNF em categorias (piloto/mecânico) | Ruan et al. [2] — RF+SHAP paper usa `driver_dnf_rate` e `constructor_dnf_rate` como features separadas |
| Documentar DNF como limitação metodológica | Arquitetura proposta, seção 2 — "Outliers legítimos como falhas mecânicas: manter com flag binária" |

A arquitetura (seção 2) menciona a flag `safety_car_flag = 1` para outliers legítimos. Para DNFs, a decisão foi exclusão completa do dataset de modelagem (não flag), seguindo Henderson et al. [9]. Essa decisão está justificada abaixo.

---

## Implementação

### Script: `src/tratamento_dnf.py`

O script recebe a base limpa da Etapa 01 e aplica dois passos: classificação e exclusão.

### Passo 1 — Critérios de registro inválido (Etapa 01)

Os seguintes critérios removem registros na etapa de limpeza, **antes** do tratamento de DNF:

| Critério | Coluna | Ação |
|---|---|---|
| `grid_position` nulo | `grid_position` | Remover |
| `finish_position` nulo | `finish_position` | Remover |
| `driver_id` nulo | `driver_id` | Remover |
| `constructor_id` nulo | `constructor_id` | Remover |
| `season` ou `round` nulo | `season`, `round` | Remover |
| RaceID duplicado | `driver_id + season + round` | Manter primeiro, remover demais |

Resultado da Etapa 01: **0 registros removidos** por nulos — a base Ergast 2018-2025 está completa nesses campos. **0 duplicatas** encontradas.

### Passo 2 — Classificação de DNF

A função `classificar_dnf()` no script analisa o campo `status` (texto livre vindo do Ergast) e atribui uma de quatro categorias:

**`classificado`** — piloto completou a corrida:
- `"Finished"` — completou todas as voltas
- `"Lapped"` — completou todas as voltas mas com atraso (1+ voltas atrás)
- Padrão `"+N Lap(s)"` — classificado com voltas atrás

**`dnf_piloto`** — abandono por incidente do piloto (7 palavras-chave):
```
accident, collision, spun off, spun-off, spin, crash, damage
```

**`dnf_carro`** — abandono por falha mecânica (33 palavras-chave):
```
engine, gearbox, transmission, clutch, hydraulics, electrical, electronics,
ers, power unit, power loss, brakes, brake, suspension, steering, radiator,
oil, water pressure, water leak, cooling system, fuel, turbo, exhaust,
mechanical, overheating, puncture, tyre, wheel, driveshaft, differential,
battery, front wing, rear wing, vibrations
```

**`dnf_outros`** — todos os demais casos (9 palavras-chave):
```
did not start, dns, withdrew, withdrawn, illness,
excluded, disqualified, retired, not classified
```

A lógica de prioridade é sequencial: primeiro verifica `classificado`, depois `dnf_piloto`, depois `dnf_carro`, por último `dnf_outros`. Status vazio ou não reconhecido cai em `dnf_outros`.

### Passo 3 — Exclusão (DNF Excluded)

Apenas registros com `dnf_categoria == "classificado"` permanecem no dataset de modelagem. Os demais são removidos. A base classificada completa é salva separadamente para rastreabilidade e para calcular `driver_dnf_rate` e `constructor_dnf_rate` na Etapa 11.

### Sobre pilotos desclassificados (`Disqualified`)

A palavra `"disqualified"` está na lista de `DNF_OUTROS_KEYWORDS`. Portanto, pilotos desclassificados recebem `dnf_categoria = "dnf_outros"` e são **excluídos** do dataset de modelagem.

Isso tem implicação prática confirmada durante a validação: na temporada 2025, Hamilton, Leclerc e Gasly foram desclassificados após o GP da China, e Norris e Piastri após Las Vegas. Esses pilotos não aparecem no fold 2025 — não por falha de extração, mas porque sua exclusão é metodologicamente correta: o modelo prediz a **posição final oficial**, e pilotos desclassificados não têm posição oficial válida.

---

## Resultados obtidos

Do `relatorio_02_tratamento_dnf.txt` (base 2018-2025):

| Categoria | Registros | % do total |
|---|---|---|
| `classificado` | 2.943 | 85,1% |
| `dnf_carro` | 174 | 5,0% |
| `dnf_piloto` | 147 | 4,3% |
| `dnf_outros` | 194 | 5,6% |
| **Total bruto** | **3.458** | 100% |

**515 registros removidos** da base de modelagem (14,9% do total bruto).

Detalhe: a diferença de `dnf_outros` entre 2018-2024 (134) e 2018-2025 (194) representa 60 registros a mais em 2025. Parte deles são as desclassificações mencionadas acima.

Base de modelagem resultante: **2.943 linhas** (confirmado em todas as etapas subsequentes).

---

## Avaliação crítica

**Por que excluir DNFs e não marcar com flag?**

Existem duas abordagens comuns na literatura:

- **DNF Included com flag**: mantém todos os registros, adiciona `is_dnf = 1` como feature. O modelo aprende a prever tanto corridas completas quanto abandonos.
- **DNF Excluded**: remove abandonos, treina apenas em corridas completas.

A escolha por DNF Excluded tem três justificativas:

1. **Target incoerente**: a posição final de um piloto que abandonou na volta 5 não mede seu desempenho — mede quando ele parou. Treinar com esse dado ensina o modelo a correlacionar features de desempenho com posições que não refletem esse desempenho.
2. **Alinhamento com o benchmark**: Henderson et al. [9] usa DNF Excluded e reporta MAE de 2.3 — a meta comparativa do TCC. Para comparar métricas, o dataset de validação precisa seguir a mesma lógica.
3. **Problema diferente**: predição de DNF é um problema de classificação separado (ocorrerá abandono mecânico?), com features próprias. Misturá-lo com predição de posição final contaminaria ambos os problemas.

**Viés de sobrevivência introduzido:**

A exclusão de DNFs cria um viés de sobrevivência documentável: o modelo aprende com a distribuição de pilotos que *completaram* a corrida. Em temporadas com alta taxa de DNF mecânico (como no início de uma nova era regulatória), esse viés pode ser relevante. Em 2026, com regulamento novo e possível instabilidade mecânica, corridas que os modelos nunca "viram" (DNF em novo contexto regulatório) podem ocorrer com mais frequência, e o modelo não terá calibrado esse risco.

**Sobre `dnf_outros` como categoria heterogênea:**

`dnf_outros` agrupa casos muito diferentes:
- `"Retired"` — abandono genérico sem causa especificada
- `"Disqualified"` — penalidade pós-corrida
- `"Did not start"` — piloto inscrito mas não largou
- `"Illness"` — causa médica

Todos são excluídos da modelagem, mas `driver_dnf_rate` e `constructor_dnf_rate` calculadas na Etapa 11 usam apenas `dnf_driver_flag` (piloto) e `dnf_car_flag` (mecânico) — `dnf_outros` não entra nas taxas de DNF. Isso é metodologicamente correto: uma desclassificação não diz nada sobre o perfil de agressividade do piloto ou a confiabilidade do carro.

---

## Convergência com a literatura

| Aspecto | Alinhado | Divergente | Observação |
|---|---|---|---|
| Variante DNF Excluded | ✅ | — | Mesma abordagem de Henderson et al. [9] |
| Classificação em piloto/mecânico/outro | ✅ | — | Necessária para calcular as features `driver_dnf_rate` e `constructor_dnf_rate` de Ruan et al. [2] |
| `Lapped` como classificado | ✅ | — | Correto — é posição oficial, apenas com atraso |
| `Disqualified` como DNF_outros → excluído | ✅ | — | Correto — sem posição oficial válida |
| Documentar viés de sobrevivência | ✅ | — | Arquitetura menciona a limitação; está explicitado aqui |
