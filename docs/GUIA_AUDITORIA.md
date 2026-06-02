# GUIA DE AUDITORIA E DOCUMENTAÇÃO — TCC F1 Predictive Model

Cada etapa é executada em ordem. Só avançamos para a próxima quando a atual estiver concluída e os arquivos entregáveis estiverem no repositório.

---

## Como usar este guia

- Marque `[x]` quando uma etapa estiver concluída.
- Cada etapa tem: **o que fazer**, **o que produzir** e **onde salvar**.
- Nenhuma etapa avança para código ou modelagem nova — o foco é documentar e validar o que já foi feito.

---

## ETAPA 1 — Inventário do Repositório

**O que fazer:**
Percorrer cada diretório e classificar cada arquivo existente.

**O que produzir:**
- `docs/inventario/inventario_dados.md` — todos os arquivos em `data/`
- `docs/inventario/inventario_scripts.md` — todos os arquivos em `src/`
- `docs/inventario/inventario_modelos.md` — todos os arquivos em `models/` e `reports/`
- `docs/inventario/inventario_documentacao.md` — todos os arquivos em `docs/`

**Classificação usada em cada tabela:**
`Essencial` | `Importante` | `Temporário` | `Candidato à remoção`

**Status:** [x] Concluído — 01/06/2026

---

## ETAPA 2 — Plano de Documentação

**O que fazer:**
Com o inventário em mãos, definir quais documentos técnicos precisam ser criados para o TCC e em que ordem.

**O que produzir:**
- `docs/plano_documentacao.md` — lista ordenada de documentos a criar, com objetivo e entregável de cada um

**Status:** [x] Concluído — 01/06/2026

---

## ETAPA 3 — Documentação Técnica (uma por etapa do pipeline)

Cada sub-etapa gera um arquivo em `docs/tecnico/`. A ordem segue o fluxo real de execução.

| # | Etapa | Arquivo gerado | Status |
|---|---|---|---|
| 3.1 | Coleta de dados | `docs/tecnico/01_coleta_dados.md` | [x] |
| 3.2 | Limpeza e DNF | `docs/tecnico/02_limpeza_dnf.md` | [x] |
| 3.3 | Encoding e normalização | `docs/tecnico/03_encoding_normalizacao.md` | [x] |
| 3.4 | Valores ausentes e outliers | `docs/tecnico/04_valores_ausentes_outliers.md` | [x] |
| 3.5 | RAPM Ridge | `docs/tecnico/05_rapm_ridge.md` | [x] |
| 3.6 | Feature Engineering | `docs/tecnico/06_feature_engineering.md` | [x] |
| 3.7 | Seleção de features (correlação + RFE) | `docs/tecnico/07_selecao_features.md` | [x] |
| 3.8 | Walk-forward e time-decay | `docs/tecnico/08_walk_forward_time_decay.md` | [x] |
| 3.9 | Modelagem e tuning | `docs/tecnico/09_modelagem_tuning.md` | [x] |
| 3.10 | Resultados e feature importance | `docs/tecnico/10_resultados_feature_importance.md` | [x] |

**Estrutura de cada arquivo:**
```
## Contexto
## Fundamentação bibliográfica
## Implementação
## Resultados obtidos
## Avaliação crítica
## Convergência com a literatura
```

---

## ETAPA 4 — Validação Metodológica

**O que fazer:**
Verificar, com evidências do código e dos dados, se o que foi implementado é metodologicamente sólido.

**O que produzir:**
- `docs/validacao_metodologica.md`

**Tópicos obrigatórios:**
- Causalidade de cada feature (evidência no código)
- Status final do leakage (safety_car_flag, weather_impact_factor)
- Reprodutibilidade (seeds, versões, manifestos)
- Cobertura dos dados (corridas por temporada vs. calendário oficial)
- Viés de sobrevivência dos DNFs

**Status:** [x] Concluído — 02/06/2026

---

## ETAPA 5 — Limpeza do Repositório

**O que fazer:**
Executar as remoções e arquivamentos identificados na etapa anterior.

**O que produzir:**
- `docs/registro_limpeza.md` — lista do que foi removido, arquivado ou mantido, com justificativa

**Regras:**
- Nada é removido sem evidência de que é regenerável ou obsoleto.
- Arquivos com papel na rastreabilidade são arquivados, não deletados.

**Status:** [x] Concluído — 02/06/2026

---

## ETAPA 6 — Roadmap para as próximas semanas

**O que fazer:**
Com toda a documentação e validação concluídas, definir o que precisa acontecer antes de avançar para a Semana 3 do cronograma.

**O que produzir:**
- `docs/roadmap.md` — resumo do estado atual + lista priorizada do que falta antes de avançar

**Status:** [x] Concluído — 02/06/2026

---

## Ordem de execução

```
Etapa 1 → Etapa 2 → Etapa 3 (3.1 → 3.2 → ... → 3.10) → Etapa 4 → Etapa 5 → Etapa 6
```

Cada etapa é concluída antes de iniciar a próxima.
Dentro da Etapa 3, cada sub-etapa é concluída antes de passar para a seguinte.

---

## Arquivos que este guia irá gerar

```
docs/
├── GUIA_AUDITORIA.md          ← este arquivo
├── inventario/
│   ├── inventario_dados.md
│   ├── inventario_scripts.md
│   ├── inventario_modelos.md
│   └── inventario_documentacao.md
├── plano_documentacao.md
├── tecnico/
│   ├── 01_coleta_dados.md
│   ├── 02_limpeza_dnf.md
│   ├── 03_encoding_normalizacao.md
│   ├── 04_valores_ausentes_outliers.md
│   ├── 05_rapm_ridge.md
│   ├── 06_feature_engineering.md
│   ├── 07_selecao_features.md
│   ├── 08_walk_forward_time_decay.md
│   ├── 09_modelagem_tuning.md
│   └── 10_resultados_feature_importance.md
├── validacao_metodologica.md
├── registro_limpeza.md
└── roadmap.md
```
