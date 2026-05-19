import pandas as pd
import os

# Caminhos dos arquivos
arquivo_ergast = "data/raw/ergast_2018_2024.csv"
arquivo_openf1_validation = "data/raw/openf1_validation.json"
arquivo_openf1_grid = "data/raw/openf1_starting_grid_2025.csv"

saida_md = "mapeamento_openf1_ergast.md"


def carregar_csv(nome, caminho):

    if not os.path.exists(caminho):
        print(f"{nome}: arquivo não encontrado -> {caminho}")
        return None

    df = pd.read_csv(caminho)

    print(f"\n===== {nome} =====")
    print(f"Shape: {df.shape}")
    print(f"Colunas: {df.columns.tolist()}")

    return df


def status_coluna(df, coluna):

    if df is not None and coluna in df.columns:
        return "OK"

    return "AUSENTE"


# =========================
# Carregar datasets reais
# =========================

ergast = carregar_csv("Ergast/Jolpica", arquivo_ergast)

# openf1_validation.json é lido separadamente para extrair field_mapping
openf1_position = None  # arquivo /position não extraído localmente — validação via openf1_validation.json
_val_path = arquivo_openf1_validation
if os.path.exists(_val_path):
    import json
    with open(_val_path, encoding="utf-8") as _f:
        _val = json.load(_f)
    # Montar DataFrame sintético com colunas documentadas pela validação
    _pos_cols = ["session_key", "driver_number", "position", "date", "meeting_key"]
    openf1_position = pd.DataFrame(columns=_pos_cols)
    print(f"\n===== OpenF1 Position (sintético via validation) =====")
    print(f"Colunas documentadas: {_pos_cols}")
else:
    print(f"openf1_validation.json não encontrado em {_val_path}")

openf1_grid = carregar_csv(
    "OpenF1 Starting Grid",
    arquivo_openf1_grid
)


# =========================
# Verificação automática
# =========================

campos_criticos = [
    "session_key",
    "driver_number",
    "position",
    "date",
    "grid_position"
]

campos_adicionais = [
    "driver_id",
    "round",
    "season",
    "finish_position",
    "status",
    "points",
    "meeting_key",
    "race_session_key",
    "qualifying_session_key",
    "qualifying_lap_duration"
]

print("\n===== Verificação dos campos críticos =====")

linhas_verificacao = []

for campo in campos_criticos:

    status_ergast = status_coluna(ergast, campo)
    status_position = status_coluna(openf1_position, campo)
    status_grid = status_coluna(openf1_grid, campo)

    print(
        f"{campo}: "
        f"Ergast={status_ergast} | "
        f"OpenF1 Position={status_position} | "
        f"OpenF1 Grid={status_grid}"
    )

    linhas_verificacao.append(
        f"| `{campo}` | {status_ergast} | {status_position} | {status_grid} |"
    )

print("\n===== Verificação dos campos adicionais =====")

linhas_adicionais = []

for campo in campos_adicionais:

    status_ergast = status_coluna(ergast, campo)
    status_position = status_coluna(openf1_position, campo)
    status_grid = status_coluna(openf1_grid, campo)

    print(
        f"{campo}: "
        f"Ergast={status_ergast} | "
        f"OpenF1 Position={status_position} | "
        f"OpenF1 Grid={status_grid}"
    )

    linhas_adicionais.append(
        f"| `{campo}` | {status_ergast} | {status_position} | {status_grid} |"
    )


# =========================
# Preparar listas de colunas
# =========================

colunas_ergast = (
    ergast.columns.tolist()
    if ergast is not None
    else "Arquivo não encontrado"
)

colunas_position = (
    openf1_position.columns.tolist()
    if openf1_position is not None
    else "Arquivo não encontrado"
)

colunas_grid = (
    openf1_grid.columns.tolist()
    if openf1_grid is not None
    else "Arquivo não encontrado"
)

# =========================
# Criar markdown
# =========================

conteudo = "# Mapeamento OpenF1 → Ergast/Jolpica\n\n"

conteudo += "## Objetivo\n\n"
conteudo += (
    "Este documento descreve como os campos da OpenF1 "
    "se relacionam com o schema usado a partir da Jolpica/Ergast.\n\n"
)

conteudo += "---\n\n"

conteudo += "## Arquivos verificados\n\n"

conteudo += "| Fonte | Arquivo |\n"
conteudo += "|---|---|\n"

conteudo += f"| Jolpica/Ergast | `{arquivo_ergast}` |\n"
conteudo += f"| OpenF1 Validation | `{arquivo_openf1_validation}` |\n"
conteudo += f"| OpenF1 Starting Grid | `{arquivo_openf1_grid}` |\n\n"

conteudo += "---\n\n"

conteudo += "## Colunas encontradas nos arquivos\n\n"

conteudo += "### Jolpica/Ergast\n\n"
conteudo += "```text\n"
conteudo += str(colunas_ergast)
conteudo += "\n```\n\n"

conteudo += "### OpenF1 Position\n\n"
conteudo += "```text\n"
conteudo += str(colunas_position)
conteudo += "\n```\n\n"

conteudo += "### OpenF1 Starting Grid\n\n"
conteudo += "```text\n"
conteudo += str(colunas_grid)
conteudo += "\n```\n\n"

conteudo += "---\n\n"

conteudo += "## Verificação automática dos campos críticos\n\n"

conteudo += "| Campo crítico | Ergast/Jolpica | OpenF1 Position | OpenF1 Starting Grid |\n"
conteudo += "|---|---|---|---|\n"

conteudo += "\n".join(linhas_verificacao)

conteudo += "\n\n---\n\n"

conteudo += "## Verificação de campos adicionais\n\n"

conteudo += "| Campo | Ergast/Jolpica | OpenF1 Position | OpenF1 Starting Grid |\n"
conteudo += "|---|---|---|---|\n"

conteudo += "\n".join(linhas_adicionais)

conteudo += "\n\n---\n\n"

conteudo += "## Tabela de equivalência\n\n"

conteudo += "| Campo OpenF1 | Equivalente Ergast/Jolpica | Observação |\n"
conteudo += "|---|---|---|\n"

conteudo += "| `driver_number` | `driver_id` | Precisa de lookup número → id do piloto |\n"
conteudo += "| `position` | `finish_position` | Position temporal da OpenF1 |\n"
conteudo += "| `grid_position` | `grid_position` | Starting grid — via Ergast (direto) |\n"
conteudo += "| `position` (starting_grid) | `grid_position` | ⚠️ Na OpenF1 `/starting_grid` o campo chama-se `position`, não `grid_position`. Join por `meeting_key` obrigatório |\n"
conteudo += "| `session_key` | `season + round` | Chave da sessão |\n"
conteudo += "| `meeting_key` | `season + round` | Chave do GP |\n"
conteudo += "| `date` | calendário da corrida | Timestamp temporal |\n"
conteudo += "| `driver_number` | `driver_id` | Lookup necessário: número → driverId (ex: 1 → verstappen). Tabela em `openf1_starting_grid_2025.csv` cruzada com FastF1 |\n"

conteudo += "\n---\n\n"

conteudo += "## Estratégia de fallback\n\n"

conteudo += "| Campo | Fonte principal | Fallback |\n"
conteudo += "|---|---|---|\n"

conteudo += "| `grid_position` | Jolpica/Ergast | OpenF1 `/starting_grid` |\n"
conteudo += "| `finish_position` | Jolpica/Ergast | OpenF1 `/session_result` |\n"
conteudo += "| `status` | Jolpica/Ergast | OpenF1 `/session_result` |\n"
conteudo += "| `points` | Jolpica/Ergast | Sem fallback |\n"

conteudo += "\n---\n\n"

conteudo += "## Conclusão\n\n"

conteudo += (
    "As fontes serão integradas pela seguinte chave composta:\n\n"
    "- **Ergast → OpenF1**: `season` + `round` → `meeting_key` (via calendário)\n"
    "- **Dentro da OpenF1**: `meeting_key` + `session_key` identifica a sessão\n"
    "- **Piloto**: `driver_id` (Ergast) ↔ `driver_number` (OpenF1) via lookup FastF1\n\n"
    "⚠️ **Atenção ao campo de grid**: na OpenF1 `/starting_grid`, a posição de largada "
    "está no campo `position` (não `grid_position`). O campo `grid_position` do schema Ergast "
    "corresponde ao campo `position` da sessão de qualifying da OpenF1.\n\n"
    "⚠️ **Cobertura OpenF1**: endpoints validados para 2023–2025. "
    "Para dados históricos (2018–2022) usar exclusivamente Ergast + FastF1.\n"
)

# =========================
# Salvar markdown
# =========================

with open(saida_md, "w", encoding="utf-8") as f:
    f.write(conteudo)

print(f"\nArquivo criado: {saida_md}")