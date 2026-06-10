from pathlib import Path
import ast
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_DIR / "data" / "raw"
SRC_DIR = ROOT_DIR / "src"

OPENF1_GRID_FILE = RAW_DIR / "openf1_starting_grid_2025.csv"
OPENF1_SESSION_RESULT_FILE = RAW_DIR / "openf1_session_result_2025_2026.csv"
OPENF1_STINTS_FILE = RAW_DIR / "openf1_stints_2025_2026.csv"
ERGAST_2025_FILE = RAW_DIR / "ergast_2025_results.csv"

SCRIPT_FILE = SRC_DIR / "update_openf1_2026.py"


def carregar_dicionario(script_path: Path, nome_dicionario: str) -> dict:
    """
    Lê automaticamente um dicionário definido no script informado.
    Exemplo: DRIVER_NUMBER_TO_ID_2025.
    """
    codigo = script_path.read_text(encoding="utf-8")
    arvore = ast.parse(codigo)

    for node in arvore.body:
        if isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == nome_dicionario:
                return ast.literal_eval(node.value)

        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == nome_dicionario:
                    return ast.literal_eval(node.value)

    raise ValueError(f"Dicionário {nome_dicionario} não encontrado em {script_path}")


def carregar_driver_numbers(caminho: Path, season: int = 2025) -> set[int]:
    df = pd.read_csv(caminho)

    if "season" in df.columns:
        df = df[df["season"] == season].copy()

    if "driver_number" not in df.columns:
        raise ValueError(f"Arquivo {caminho.name} não possui coluna driver_number.")

    return set(df["driver_number"].dropna().astype(int).unique())


def main():
    driver_number_to_id_2025 = carregar_dicionario(
        SCRIPT_FILE,
        "DRIVER_NUMBER_TO_ID_2025"
    )

    ergast_2025 = pd.read_csv(ERGAST_2025_FILE)

    ids_ergast_2025 = set(
        ergast_2025["driver_id"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
    )

    ids_mapeados = set(driver_number_to_id_2025.values())
    numeros_mapeados = set(driver_number_to_id_2025.keys())

    arquivos_openf1 = {
        "openf1_starting_grid_2025.csv": OPENF1_GRID_FILE,
        "openf1_session_result_2025_2026.csv": OPENF1_SESSION_RESULT_FILE,
        "openf1_stints_2025_2026.csv": OPENF1_STINTS_FILE,
    }

    print("\n=== VERIFICAÇÃO DO MAPEAMENTO OPENF1 → ERGAST/JOLPICA ===\n")

    print(f"Entradas no DRIVER_NUMBER_TO_ID_2025: {len(driver_number_to_id_2025)}")
    print(f"driver_id únicos no Ergast/Jolpica 2025: {len(ids_ergast_2025)}")

    print("\n--- Comparação com arquivos OpenF1 2025 ---")

    todos_numeros_openf1 = set()

    for nome, caminho in arquivos_openf1.items():
        if not caminho.exists():
            print(f"{nome}: arquivo não encontrado")
            continue

        numeros_openf1 = carregar_driver_numbers(caminho, season=2025)
        todos_numeros_openf1.update(numeros_openf1)

        sem_mapeamento = sorted(numeros_openf1 - numeros_mapeados)
        mapeados_nao_usados = sorted(numeros_mapeados - numeros_openf1)

        print(f"\nArquivo: {nome}")
        print(f"driver_number únicos OpenF1 2025: {len(numeros_openf1)}")
        print(f"driver_number sem mapeamento: {sem_mapeamento}")
        print(f"driver_number mapeados que não aparecem nesse arquivo: {mapeados_nao_usados}")

    numeros_openf1_sem_mapeamento = sorted(todos_numeros_openf1 - numeros_mapeados)
    numeros_mapeados_sem_openf1 = sorted(numeros_mapeados - todos_numeros_openf1)

    ids_mapeados_sem_ergast = sorted(ids_mapeados - ids_ergast_2025)
    ids_ergast_sem_mapeamento = sorted(ids_ergast_2025 - ids_mapeados)

    print("\n--- Consolidação 2025 ---")
    print(f"driver_number únicos em todos os arquivos OpenF1 2025: {len(todos_numeros_openf1)}")
    print(f"driver_number OpenF1 2025 sem mapeamento: {numeros_openf1_sem_mapeamento}")
    print(f"driver_number mapeados que não aparecem na OpenF1 2025: {numeros_mapeados_sem_openf1}")

    print("\n--- Comparação dos driver_id ---")
    print(f"driver_id mapeados sem existir no Ergast/Jolpica 2025: {ids_mapeados_sem_ergast}")
    print(f"driver_id Ergast/Jolpica 2025 sem mapeamento: {ids_ergast_sem_mapeamento}")

    if (
        len(numeros_openf1_sem_mapeamento) == 0
        and len(ids_mapeados_sem_ergast) == 0
        and len(ids_ergast_sem_mapeamento) == 0
    ):
        print("\nRESULTADO: mapeamento OpenF1 → Ergast/Jolpica correto para 2025.")
    else:
        print("\nRESULTADO: existem inconsistências no mapeamento OpenF1 → Ergast/Jolpica.")


if __name__ == "__main__":
    main()