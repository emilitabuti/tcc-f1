from __future__ import annotations

from datetime import datetime
from pathlib import Path

import json
import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# update_openf1_2026.py
#
# Objetivo:
#   Extrair e processar as corridas de 2026 já disponíveis nos arquivos raw
#   e via API OpenF1, aplicando o mesmo schema de features do pipeline
#   2018-2025 para permitir a análise de drift (Semana 3, P2).
#
# Saídas:
#   data/processed/openf1_2026_available.csv  — X com as 15 features + y
#   data/processed/relatorio_update_2026.txt  — relatório de cobertura
#
# Como usar:
#   python src/update_openf1_2026.py
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models" / "feature_selection"

OUTPUT_CSV = PROCESSED_DIR / "openf1_2026_available.csv"
OUTPUT_REPORT = PROCESSED_DIR / "relatorio_update_2026.txt"

OPENF1_BASE = "https://api.openf1.org/v1"

# Features exatas do contrato — mesma ordem do dataset_modelagem_X_2018_2025.csv
FEATURES_FINAIS = [
    "qualifying_position",
    "grid_penalty",
    "recent_form_5",
    "driver_coef_rapm",
    "driver_dnf_rate",
    "constructor_coef_rapm",
    "constructor_dnf_rate",
    "constructor_wins_total",
    "driver_constructor_synergy",
    "track_complexity",
    "altitude_m",
    "tire_compound_start",
    "avg_pit_stops_circuit",
    "season_factor",
    "incident_rate_hist_norm",
]

KEY_COLS = ["RaceID", "season", "round", "race_name", "driver_id", "constructor_id"]
TARGET = "finish_position"

# ---------------------------------------------------------------------------
# Mapeamento driver_number -> driver_id (construído a partir de Abu Dhabi 2025)
# Atualizar manualmente quando novos pilotos 2026 forem confirmados.
# ---------------------------------------------------------------------------
DRIVER_NUMBER_TO_ID_2025: dict[int, str] = {
    1: "max_verstappen",
    4: "norris",
    5: "bortoleto",
    6: "hadjar",
    7: "doohan",
    10: "gasly",
    12: "antonelli",
    14: "alonso",
    16: "leclerc",
    18: "stroll",
    22: "tsunoda",
    23: "albon",
    27: "hulkenberg",
    30: "lawson",
    31: "ocon",
    43: "colapinto",
    44: "hamilton",
    55: "sainz",
    63: "russell",
    81: "piastri",
    87: "bearman",
}

# Pilotos novos em 2026 (números 3, 11, 41, 77 — Cadillac e mudanças de número).
# Preencher com os IDs reais quando confirmados.
DRIVER_NUMBER_TO_ID_2026_NEW: dict[int, str] = {
    3: "driver_3_2026",
    11: "driver_11_2026",
    41: "driver_41_2026",
    77: "driver_77_2026",
}

DRIVER_NUMBER_TO_ID: dict[int, str] = {
    **DRIVER_NUMBER_TO_ID_2025,
    **DRIVER_NUMBER_TO_ID_2026_NEW,
}


# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

def _get(endpoint: str, params: dict) -> list[dict]:
    url = f"{OPENF1_BASE}/{endpoint}"
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        print(f"  [AVISO] OpenF1 API falhou: {exc}")
        return []


def carregar_meetings_2026() -> pd.DataFrame:
    meetings = pd.read_csv(RAW_DIR / "openf1_meetings_2025_2026.csv")
    m2026 = meetings[
        (meetings["season"] == 2026)
        & (~meetings["meeting_name"].str.contains("Testing", case=False, na=False))
    ].copy()
    return m2026


def carregar_resultados_2026() -> pd.DataFrame:
    results = pd.read_csv(RAW_DIR / "openf1_session_result_2025_2026.csv")
    return results[results["season"] == 2026].copy()


def carregar_stints_2026() -> pd.DataFrame:
    stints = pd.read_csv(RAW_DIR / "openf1_stints_2025_2026.csv")
    return stints[stints["season"] == 2026].copy() if "season" in stints.columns else pd.DataFrame()


def carregar_race_control_2026() -> pd.DataFrame:
    rc = pd.read_csv(RAW_DIR / "openf1_race_control_2025_2026.csv")
    return rc[rc["season"] == 2026].copy() if "season" in rc.columns else pd.DataFrame()


def buscar_qualifying_openf1(meeting_key: int) -> pd.DataFrame:
    """
    Busca posições de qualifying via API OpenF1.
    Segue o mesmo padrão do extract_openf1_starting_grid_2025.py:
      1. GET /sessions?meeting_key=X → encontra session_key da Qualifying
      2. GET /starting_grid?session_key=qualifying_session_key → posições
    """
    sessions = _get("sessions", {"meeting_key": meeting_key})
    if not sessions:
        return pd.DataFrame()

    df_sessions = pd.DataFrame(sessions)
    qual = df_sessions[df_sessions.get("session_name", pd.Series()).eq("Qualifying")] \
        if "session_name" in df_sessions.columns else pd.DataFrame()

    if qual.empty:
        print(f"    [AVISO] Sem sessão de Qualifying para meeting_key={meeting_key}")
        return pd.DataFrame()

    qualifying_session_key = qual.iloc[0]["session_key"]
    grid = _get("starting_grid", {"session_key": qualifying_session_key})
    if not grid:
        return pd.DataFrame()

    df = pd.DataFrame(grid)
    df["meeting_key"] = meeting_key
    # Renomear position → qualifying_position para clareza
    if "position" in df.columns:
        df = df.rename(columns={"position": "qualifying_position"})
    return df


def mapear_driver_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["driver_id"] = df["driver_number"].map(DRIVER_NUMBER_TO_ID)
    n_unknown = df["driver_id"].isna().sum()
    if n_unknown > 0:
        unknown_nums = df[df["driver_id"].isna()]["driver_number"].unique()
        print(f"  [AVISO] {n_unknown} linhas com driver_number sem mapeamento: {list(unknown_nums)}")
        df["driver_id"] = df["driver_id"].fillna(
            df["driver_number"].apply(lambda n: f"driver_{int(n)}_2026")
        )
    return df


def carregar_features_historicas() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega os datasets finais 2018-2025 para extrair features históricas."""
    X = pd.read_csv(PROCESSED_DIR / "dataset_modelagem_X_2018_2025.csv")
    y = pd.read_csv(PROCESSED_DIR / "dataset_modelagem_y_2018_2025.csv")
    return X, y


def snapshot_features_fim_2025(X: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    """
    Extrai um snapshot das features históricas no final de 2025 por piloto/construtor.
    Usado como ponto de partida para as features de forma recente e acumulados.
    """
    df = pd.concat([y[["RaceID", "season", "round", "driver_id", "constructor_id"]], X], axis=1)
    df_2025 = df[df["season"] == 2025].copy()

    # Última corrida de cada piloto em 2025
    idx_last = df_2025.groupby("driver_id")["round"].idxmax()
    snapshot_drivers = df_2025.loc[idx_last].set_index("driver_id")[
        ["driver_coef_rapm", "driver_dnf_rate", "recent_form_5",
         "driver_constructor_synergy", "constructor_id"]
    ].copy()

    # Última corrida de cada construtor em 2025
    idx_last_c = df_2025.groupby("constructor_id")["round"].idxmax()
    snapshot_constructors = df_2025.loc[idx_last_c].set_index("constructor_id")[
        ["constructor_coef_rapm", "constructor_dnf_rate", "constructor_wins_total"]
    ].copy()

    return snapshot_drivers, snapshot_constructors


def carregar_features_circuito() -> pd.DataFrame:
    """
    Carrega features fixas de circuito do dataset 2018-2025.
    """
    X = pd.read_csv(PROCESSED_DIR / "dataset_modelagem_X_2018_2025.csv")
    y = pd.read_csv(PROCESSED_DIR / "dataset_modelagem_y_2018_2025.csv")
    df = pd.concat([y[["season", "round", "driver_id"]], X], axis=1)

    circuit_cols = [
        "track_complexity", "altitude_m", "avg_pit_stops_circuit", "incident_rate_hist_norm"
    ]
    # Uma linha por round (features de circuito são idênticas para todos os pilotos)
    circuito = df.groupby("round")[circuit_cols].first().reset_index()
    return circuito


def inferir_compound_ordinal(stints_2026: pd.DataFrame, meeting_key: int) -> pd.Series:
    """Infere compound de largada a partir dos stints."""
    if stints_2026.empty or "meeting_key" not in stints_2026.columns:
        return pd.Series(dtype=float)

    corrida = stints_2026[stints_2026["meeting_key"] == meeting_key].copy()
    if corrida.empty:
        return pd.Series(dtype=float)

    # Stint 1 = composto de largada
    stint1 = corrida[corrida["stint_number"] == 1] if "stint_number" in corrida.columns else corrida

    compound_map = {"SOFT": 1, "MEDIUM": 2, "HARD": 3, "INTERMEDIATE": 4, "WET": 5}
    compound_col = [c for c in stint1.columns if "compound" in c.lower()]
    if not compound_col:
        return pd.Series(dtype=float)

    stint1 = stint1.copy()
    stint1["tire_compound_start"] = stint1[compound_col[0]].str.upper().map(compound_map).fillna(2)

    driver_col = "driver_number" if "driver_number" in stint1.columns else None
    if driver_col:
        return stint1.set_index(driver_col)["tire_compound_start"]
    return pd.Series(dtype=float)


def detectar_safety_car(race_control_2026: pd.DataFrame, meeting_key: int) -> int:
    if race_control_2026.empty or "meeting_key" not in race_control_2026.columns:
        return 0
    rc = race_control_2026[race_control_2026["meeting_key"] == meeting_key]
    if "category" not in rc.columns:
        return 0
    return int(rc["category"].str.contains("SafetyCar", case=False, na=False).any())


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def processar_corrida_2026(
    meeting_key: int,
    race_name: str,
    round_num: int,
    results_2026: pd.DataFrame,
    stints_2026: pd.DataFrame,
    rc_2026: pd.DataFrame,
    snapshot_drivers: pd.DataFrame,
    snapshot_constructors: pd.DataFrame,
    circuito_features: pd.DataFrame,
    log: list[str],
) -> pd.DataFrame:

    print(f"\n  Processando Round {round_num} — {race_name} (meeting_key={meeting_key})")

    # 1. Resultado da corrida
    corrida_raw = results_2026[results_2026["meeting_key"] == meeting_key].copy()
    if corrida_raw.empty:
        log.append(f"Round {round_num} ({race_name}): SEM RESULTADO — pulado")
        return pd.DataFrame()

    corrida_raw = mapear_driver_ids(corrida_raw)
    corrida_raw = corrida_raw.rename(columns={"position": "finish_position"})
    corrida_raw["is_dnf"] = corrida_raw["dnf"].astype(bool) if "dnf" in corrida_raw.columns else False

    # Excluir DNFs (mesmo critério do pipeline 2018-2025)
    corrida = corrida_raw[~corrida_raw["is_dnf"]].copy()
    n_total = len(corrida_raw)
    n_finishers = len(corrida)
    log.append(
        f"Round {round_num} ({race_name}): {n_total} pilotos raw, "
        f"{n_finishers} após exclusão de DNF"
    )

    if corrida.empty:
        return pd.DataFrame()

    # 2. Qualifying position via OpenF1 API
    print(f"    Buscando qualifying via OpenF1 API...")
    qual_df = buscar_qualifying_openf1(meeting_key)

    if not qual_df.empty and "driver_number" in qual_df.columns and "qualifying_position" in qual_df.columns:
        qual_df = mapear_driver_ids(qual_df)
        qual_map = qual_df.set_index("driver_id")["qualifying_position"].to_dict()
        corrida["qualifying_position"] = corrida["driver_id"].map(qual_map)
        n_qual = corrida["qualifying_position"].notna().sum()
        log.append(f"  qualifying_position: {n_qual}/{n_finishers} preenchidos via API")
        print(f"    qualifying_position via API: {n_qual}/{n_finishers}")
    else:
        # Fallback: grid_position não disponível → usar mediana histórica da corrida
        corrida["qualifying_position"] = np.nan
        log.append(f"  qualifying_position: API sem dados — usando NaN (requer preenchimento manual)")
        print(f"    [AVISO] qualifying_position não disponível via API")

    # grid_penalty = 0 como default (sem informação de penalidades de grid)
    corrida["grid_penalty"] = 0

    # 3. Features históricas do piloto (snapshot fim de 2025)
    corrida["driver_coef_rapm"] = corrida["driver_id"].map(
        snapshot_drivers["driver_coef_rapm"]
    ).fillna(0.0)

    corrida["driver_dnf_rate"] = corrida["driver_id"].map(
        snapshot_drivers["driver_dnf_rate"]
    ).fillna(0.0)

    corrida["recent_form_5"] = corrida["driver_id"].map(
        snapshot_drivers["recent_form_5"]
    ).fillna(0.0)

    corrida["driver_constructor_synergy"] = corrida["driver_id"].map(
        snapshot_drivers["driver_constructor_synergy"]
    ).fillna(0.0)

    # constructor_id: do snapshot de 2025 ou fallback por driver_number
    corrida["constructor_id"] = corrida["driver_id"].map(
        snapshot_drivers["constructor_id"]
    ).fillna("unknown_2026")

    # 4. Features históricas do construtor
    corrida["constructor_coef_rapm"] = corrida["constructor_id"].map(
        snapshot_constructors["constructor_coef_rapm"]
    ).fillna(0.0)

    corrida["constructor_dnf_rate"] = corrida["constructor_id"].map(
        snapshot_constructors["constructor_dnf_rate"]
    ).fillna(0.0)

    corrida["constructor_wins_total"] = corrida["constructor_id"].map(
        snapshot_constructors["constructor_wins_total"]
    ).fillna(0.0)

    # 5. Features de circuito (lookup por round)
    circ = circuito_features[circuito_features["round"] == round_num]
    if not circ.empty:
        for col in ["track_complexity", "altitude_m", "avg_pit_stops_circuit", "incident_rate_hist_norm"]:
            corrida[col] = circ.iloc[0][col]
    else:
        # Circuito novo em 2026 sem histórico — usar mediana global
        for col in ["track_complexity", "altitude_m", "avg_pit_stops_circuit", "incident_rate_hist_norm"]:
            corrida[col] = np.nan
        log.append(f"  circuito: round {round_num} sem histórico — features de circuito como NaN")

    # 6. Composto de largada via stints
    compound_series = inferir_compound_ordinal(stints_2026, meeting_key)
    if not compound_series.empty:
        corrida["tire_compound_start"] = corrida["driver_number"].map(compound_series).fillna(2.0)
    else:
        corrida["tire_compound_start"] = 2.0  # fallback: MEDIUM

    # 7. Season factor
    corrida["season_factor"] = 2026

    # 8. Safety car (auditoria — não entra em X)
    corrida["safety_car_flag"] = detectar_safety_car(rc_2026, meeting_key)

    # 9. Chaves
    corrida["season"] = 2026
    corrida["round"] = round_num
    corrida["race_name"] = race_name
    corrida["RaceID"] = (
        corrida["driver_id"].astype(str)
        + "_2026_"
        + corrida["round"].astype(str)
    )

    return corrida


def main():
    print("=" * 60)
    print("update_openf1_2026.py")
    print(f"Executado em: {datetime.now().isoformat(timespec='seconds')}")
    print("=" * 60)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    log: list[str] = [
        f"Relatório — update_openf1_2026.py",
        f"Gerado em: {datetime.now().isoformat(timespec='seconds')}",
        "",
    ]

    # Carregar dados raw 2026
    print("\n[1] Carregando arquivos raw 2026...")
    meetings_2026 = carregar_meetings_2026()
    results_2026 = carregar_resultados_2026()
    stints_2026 = carregar_stints_2026()
    rc_2026 = carregar_race_control_2026()

    meetings_com_resultado = set(results_2026["meeting_key"].unique())
    meetings_disponiveis = meetings_2026[
        meetings_2026["meeting_key"].isin(meetings_com_resultado)
    ].copy()

    print(f"  Meetings 2026 no calendário: {len(meetings_2026)}")
    print(f"  Meetings 2026 com resultado: {len(meetings_disponiveis)}")
    log.append(f"Meetings 2026 no calendário: {len(meetings_2026)}")
    log.append(f"Meetings 2026 com resultado: {len(meetings_disponiveis)}")
    log.append("")

    # Carregar features históricas
    print("\n[2] Carregando features históricas do pipeline 2018-2025...")
    X, y = carregar_features_historicas()
    snapshot_drivers, snapshot_constructors = snapshot_features_fim_2025(X, y)
    circuito_features = carregar_features_circuito()
    print(f"  Snapshot drivers: {len(snapshot_drivers)} entidades")
    print(f"  Snapshot construtores: {len(snapshot_constructors)} entidades")

    # Construir mapeamento round → meeting_key via calendário 2026
    # Os rounds 2026 são atribuídos por ordem cronológica de meeting_key
    meetings_2026_sorted = meetings_2026.sort_values("meeting_key").reset_index(drop=True)
    meetings_2026_sorted["round"] = range(1, len(meetings_2026_sorted) + 1)
    mk_to_round = meetings_2026_sorted.set_index("meeting_key")["round"].to_dict()
    mk_to_name = meetings_2026_sorted.set_index("meeting_key")["meeting_name"].to_dict()

    # Processar cada corrida disponível
    print("\n[3] Processando corridas 2026 disponíveis...")
    frames: list[pd.DataFrame] = []

    for _, row in meetings_disponiveis.iterrows():
        mk = int(row["meeting_key"])
        rnd = mk_to_round.get(mk, 0)
        name = mk_to_name.get(mk, row["meeting_name"])

        df_corrida = processar_corrida_2026(
            meeting_key=mk,
            race_name=name,
            round_num=rnd,
            results_2026=results_2026,
            stints_2026=stints_2026,
            rc_2026=rc_2026,
            snapshot_drivers=snapshot_drivers,
            snapshot_constructors=snapshot_constructors,
            circuito_features=circuito_features,
            log=log,
        )

        if not df_corrida.empty:
            frames.append(df_corrida)

    if not frames:
        print("\n[ERRO] Nenhuma corrida 2026 processada.")
        return

    df_final = pd.concat(frames, ignore_index=True)

    # Garantir que as 15 features estão presentes
    missing_features = [f for f in FEATURES_FINAIS if f not in df_final.columns]
    if missing_features:
        print(f"\n[AVISO] Features ausentes no output: {missing_features}")
        log.append(f"Features ausentes: {missing_features}")
        for f in missing_features:
            df_final[f] = np.nan

    # Estatísticas de cobertura
    print("\n[4] Validação do dataset gerado...")
    log.append("")
    log.append("=== Cobertura das features ===")
    for f in FEATURES_FINAIS:
        n_nulos = int(df_final[f].isna().sum())
        pct = n_nulos / len(df_final) * 100 if len(df_final) > 0 else 0
        status = "OK" if n_nulos == 0 else f"ATENÇÃO: {n_nulos} nulos ({pct:.1f}%)"
        log.append(f"  {f}: {status}")
        print(f"  {f}: {status}")

    # Salvar
    colunas_saida = KEY_COLS + [TARGET, "is_dnf", "safety_car_flag"] + FEATURES_FINAIS
    colunas_saida = [c for c in colunas_saida if c in df_final.columns]
    df_saida = df_final[colunas_saida].sort_values(["round", "finish_position"]).reset_index(drop=True)

    df_saida.to_csv(OUTPUT_CSV, index=False)

    log.append("")
    log.append("=== Resumo final ===")
    log.append(f"Corridas processadas: {df_final['round'].nunique()}")
    log.append(f"Linhas totais: {len(df_saida)}")
    log.append(f"Pilotos únicos: {df_saida['driver_id'].nunique()}")
    log.append(f"NaN em qualifying_position: {int(df_saida['qualifying_position'].isna().sum())}")
    log.append(f"Output: {OUTPUT_CSV}")

    OUTPUT_REPORT.write_text("\n".join(log), encoding="utf-8")

    print(f"\n[CONCLUÍDO]")
    print(f"  Corridas: {df_final['round'].nunique()}")
    print(f"  Linhas: {len(df_saida)}")
    print(f"  Output: {OUTPUT_CSV}")
    print(f"  Relatório: {OUTPUT_REPORT}")
    print()
    if df_saida["qualifying_position"].isna().any():
        print("[ATENÇÃO] qualifying_position tem NaN — API OpenF1 não retornou dados de qualifying.")
        print("  Opções:")
        print("  1. Preencher manualmente com os resultados de qualifying de cada GP.")
        print("  2. Usar grid_position como proxy (se disponível via outra fonte).")


if __name__ == "__main__":
    main()
