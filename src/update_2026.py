from __future__ import annotations

import logging
import os
import socket
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# monkey-patch pra redirecionar o FastF1 pra Jolpica em vez da Ergast original
import fastf1.ergast.interface as _ff1_ergast
import fastf1.ergast.legacy as _ff1_legacy
_ff1_ergast.BASE_URL = "https://api.jolpi.ca/ergast/f1"
_ff1_legacy.base_url = "https://api.jolpi.ca/ergast/f1"

import fastf1

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CACHE_DIR = RAW_DIR / "fastf1_cache"

OUTPUT_CSV = PROCESSED_DIR / "fastf1_2026_available.csv"
OUTPUT_REPORT = PROCESSED_DIR / "relatorio_update_2026.txt"

CACHE_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

fastf1.Cache.enable_cache(str(CACHE_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)
logging.getLogger("fastf1").setLevel(logging.WARNING)

# contrato de features - tem que ser exatamente esse conjunto, igual ao pipeline 2018-2025
FEATURES_FINAIS = [
    "qualifying_position",
    "constructor_coef_rapm",
    "recent_form_5",
    "driver_constructor_synergy",
    "constructor_wins_total",
    "driver_coef_rapm",
    "track_complexity",
    "tire_compound_start",
    "season_factor",
    "avg_pit_stops_circuit",
    "constructor_dnf_rate",
    "grid_penalty",
    "altitude_m",
]

KEY_COLS = ["RaceID", "season", "round", "race_name", "driver_id", "constructor_id"]
TARGET = "finish_position"

# tabela de código 3 letras -> driver_id do Ergast, mesma do script de integração
DRIVER_CODE_TO_ID: dict[str, str] = {
    "AIT": "aitken",          "ALB": "albon",           "ALO": "alonso",
    "ANT": "antonelli",       "BEA": "bearman",         "BOR": "bortoleto",
    "BOT": "bottas",          "COL": "colapinto",       "DEV": "de_vries",
    "DOO": "doohan",          "ERI": "ericsson",        "FIT": "pietro_fittipaldi",
    "GAS": "gasly",           "GIO": "giovinazzi",      "GRO": "grosjean",
    "HAD": "hadjar",          "HAM": "hamilton",        "HAR": "brendon_hartley",
    "HUL": "hulkenberg",      "KUB": "kubica",          "KVY": "kvyat",
    "LAT": "latifi",          "LAW": "lawson",          "LEC": "leclerc",
    "MAG": "kevin_magnussen", "MAZ": "mazepin",         "MSC": "mick_schumacher",
    "NOR": "norris",          "OCO": "ocon",            "PER": "perez",
    "PIA": "piastri",         "RAI": "raikkonen",       "RIC": "ricciardo",
    "RUS": "russell",         "SAI": "sainz",           "SAR": "sargeant",
    "SIR": "sirotkin",        "STR": "stroll",          "TSU": "tsunoda",
    "VAN": "vandoorne",       "VER": "max_verstappen",  "VET": "vettel",
    "ZHO": "zhou",
    # Pilotos novos 2026 - atualizar quando os códigos oficiais forem confirmados
    "CAD": "cadillac_driver1_2026",
    "CAD2": "cadillac_driver2_2026",
}

COMPOUND_MAP: dict[str, int] = {
    "SOFT": 1, "MEDIUM": 2, "HARD": 3, "INTERMEDIATE": 4, "WET": 5,
}


MAX_RETRIES = 3
RETRY_BACKOFF = 10  # segundos; dobra a cada tentativa


def _is_no_data(exc: Exception) -> bool:
    msg = str(exc)
    return "has not been loaded yet" in msg or "session.laps vazio" in msg


def _load_session(event, stype: str, year: int, round_num: int,
                  laps: bool = True, telemetry: bool = False,
                  weather: bool = False):
    # tenta carregar a sessão com retry pra não quebrar à toa
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            session = event.get_session(stype)
            socket.setdefaulttimeout(90)
            session.load(laps=laps, telemetry=telemetry, weather=weather, messages=False)
            socket.setdefaulttimeout(None)
            return session
        except Exception as exc:
            last_exc = exc
            if _is_no_data(exc):
                raise RuntimeError("no_data") from exc
            wait = RETRY_BACKOFF * (2 ** (attempt - 1))
            log.warning(
                f"  Tentativa {attempt}/{MAX_RETRIES} ({year} R{round_num} {stype}): "
                f"{exc}. Aguardando {wait}s..."
            )
            time.sleep(wait)
    raise RuntimeError(f"Falha após {MAX_RETRIES} tentativas") from last_exc


def extrair_resultados(race_session) -> pd.DataFrame:
    # extrai finish_position e is_dnf de session.results
    res = race_session.results
    cols = [c for c in ["Abbreviation", "Position", "Status"] if c in res.columns]
    if not cols or "Position" not in cols:
        return pd.DataFrame()

    df = res[cols].copy().rename(columns={"Abbreviation": "driver_code"})
    df["finish_position"] = pd.to_numeric(df["Position"], errors="coerce")

    if "Status" in df.columns:
        # classificado = terminou ou completou voltas a menos; qualquer outro é DNF
        df["is_dnf"] = ~df["Status"].str.match(
            r"^(Finished|\+\d+ Lap)", case=False, na=True
        )
    else:
        df["is_dnf"] = df["finish_position"].isna()

    # converte código pra driver_id, usa fallback se não tiver no mapa
    df["driver_id"] = df["driver_code"].map(DRIVER_CODE_TO_ID)
    n_unknown = df["driver_id"].isna().sum()
    if n_unknown > 0:
        unknown = df[df["driver_id"].isna()]["driver_code"].unique().tolist()
        log.warning(f"  Códigos sem mapeamento: {unknown} -> usando código como fallback")
        df["driver_id"] = df["driver_id"].fillna(
            df["driver_code"].apply(lambda c: f"driver_{c}_2026")
        )

    return df[["driver_id", "driver_code", "finish_position", "is_dnf"]]


def extrair_qualifying(qual_session) -> pd.Series:
    # retorna Series driver_id -> qualifying_position
    res = qual_session.results
    if res.empty or "Abbreviation" not in res.columns or "Position" not in res.columns:
        return pd.Series(dtype=float)

    df = res[["Abbreviation", "Position"]].copy()
    df["driver_id"] = df["Abbreviation"].map(DRIVER_CODE_TO_ID).fillna(
        df["Abbreviation"].apply(lambda c: f"driver_{c}_2026")
    )
    df["qualifying_position"] = pd.to_numeric(df["Position"], errors="coerce")
    return df.set_index("driver_id")["qualifying_position"].dropna()


def extrair_compound_largada(race_session) -> pd.Series:
    # retorna Series driver_id -> tire_compound_start (ordinal 1-5)
    laps = race_session.laps
    if laps is None or laps.empty:
        return pd.Series(dtype=float)

    cols = [c for c in ["Driver", "Stint", "Compound"] if c in laps.columns]
    if len(cols) < 3:
        return pd.Series(dtype=float)

    # pega so o stint 1 pra saber o pneu de largada
    stint1 = laps[laps["Stint"] == 1].groupby("Driver")["Compound"].first()
    result = stint1.str.upper().map(COMPOUND_MAP).fillna(2.0)
    result.index = result.index.map(
        lambda code: DRIVER_CODE_TO_ID.get(code, f"driver_{code}_2026")
    )
    return result


def extrair_safety_car(race_session) -> int:
    # retorna 1 se houve SC/VSC, 0 caso contrário
    laps = race_session.laps
    if laps is None or laps.empty or "TrackStatus" not in laps.columns:
        return 0
    ts = laps["TrackStatus"].astype(str)
    # status 4 = SC, 6 = VSC, 7 = VSC ending
    return int(
        ts.str.contains("4", regex=False).any()
        | ts.str.contains("6", regex=False).any()
        | ts.str.contains("7", regex=False).any()
    )


def carregar_features_historicas():
    X = pd.read_csv(PROCESSED_DIR / "dataset_modelagem_X_2018_2025.csv")
    y = pd.read_csv(PROCESSED_DIR / "dataset_modelagem_y_2018_2025.csv")
    return X, y


def snapshot_features_fim_2025(X: pd.DataFrame, y: pd.DataFrame):
    # une X e y pra ter os metadados junto, filtra só 2025
    df = pd.concat([y[["RaceID", "season", "round", "driver_id", "constructor_id"]], X], axis=1)
    df_2025 = df[df["season"] == 2025].copy()

    # pega o estado mais recente de cada piloto no fim de 2025
    idx_last = df_2025.groupby("driver_id")["round"].idxmax()
    driver_cols = ["driver_coef_rapm", "recent_form_5", "driver_constructor_synergy", "constructor_id"]
    snapshot_drivers = df_2025.loc[idx_last].set_index("driver_id")[driver_cols].copy()
    if "driver_dnf_rate" not in snapshot_drivers.columns:
        snapshot_drivers["driver_dnf_rate"] = 0.0

    # mesma coisa pra construtores
    idx_last_c = df_2025.groupby("constructor_id")["round"].idxmax()
    snapshot_constructors = df_2025.loc[idx_last_c].set_index("constructor_id")[
        ["constructor_coef_rapm", "constructor_dnf_rate", "constructor_wins_total"]
    ].copy()

    return snapshot_drivers, snapshot_constructors


def carregar_features_circuito() -> pd.DataFrame:
    X = pd.read_csv(PROCESSED_DIR / "dataset_modelagem_X_2018_2025.csv")
    y = pd.read_csv(PROCESSED_DIR / "dataset_modelagem_y_2018_2025.csv")
    df = pd.concat([y[["season", "round", "driver_id"]], X], axis=1)

    # pega as features de circuito agrupadas por round - são iguais pra todos os pilotos
    circuit_cols = [
        col for col in ["track_complexity", "altitude_m", "avg_pit_stops_circuit", "incident_rate_hist_norm"]
        if col in df.columns
    ]
    return df.groupby("round")[circuit_cols].first().reset_index()


def processar_corrida(
    event,
    round_num: int,
    race_name: str,
    snapshot_drivers: pd.DataFrame,
    snapshot_constructors: pd.DataFrame,
    circuito_features: pd.DataFrame,
    log_lines: list[str],
) -> pd.DataFrame:

    log.info(f"  Round {round_num} - {race_name}")

    # carrega a sessão de corrida
    try:
        race_session = _load_session(event, "R", 2026, round_num, laps=True)
    except RuntimeError as exc:
        msg = f"Round {round_num} ({race_name}): falha ao carregar sessão - {exc}"
        log.warning(f"  {msg}")
        log_lines.append(msg)
        return pd.DataFrame()

    resultados = extrair_resultados(race_session)
    if resultados.empty:
        msg = f"Round {round_num} ({race_name}): SEM RESULTADO - pulado"
        log.warning(f"  {msg}")
        log_lines.append(msg)
        return pd.DataFrame()

    n_total = len(resultados)
    # joga fora os DNFs, só queremos quem completou a corrida
    corrida = resultados[~resultados["is_dnf"]].copy()
    n_finishers = len(corrida)
    log_lines.append(
        f"Round {round_num} ({race_name}): {n_total} pilotos raw, "
        f"{n_finishers} após exclusão de DNF"
    )

    if corrida.empty:
        return pd.DataFrame()

    compound_series = extrair_compound_largada(race_session)
    safety_car = extrair_safety_car(race_session)

    # carrega qualifying - se não tiver, fica NaN mesmo
    try:
        qual_session = _load_session(event, "Q", 2026, round_num, laps=False)
        qual_map = extrair_qualifying(qual_session)
    except RuntimeError:
        qual_map = pd.Series(dtype=float)

    corrida["qualifying_position"] = corrida["driver_id"].map(qual_map)
    n_qual = corrida["qualifying_position"].notna().sum()
    log_lines.append(f"  qualifying_position: {n_qual}/{n_finishers} preenchidos via FastF1")
    log.info(f"    qualifying_position: {n_qual}/{n_finishers}")

    corrida["grid_penalty"] = 0

    # features históricas do piloto - pega do snapshot fim 2025
    corrida["driver_coef_rapm"] = corrida["driver_id"].map(snapshot_drivers["driver_coef_rapm"]).fillna(0.0)
    corrida["driver_dnf_rate"] = corrida["driver_id"].map(snapshot_drivers.get("driver_dnf_rate", pd.Series(dtype=float))).fillna(0.0)
    corrida["recent_form_5"] = corrida["driver_id"].map(snapshot_drivers["recent_form_5"]).fillna(0.0)
    corrida["driver_constructor_synergy"] = corrida["driver_id"].map(snapshot_drivers["driver_constructor_synergy"]).fillna(0.0)
    corrida["constructor_id"] = corrida["driver_id"].map(snapshot_drivers["constructor_id"]).fillna("unknown_2026")

    # features do construtor
    corrida["constructor_coef_rapm"] = corrida["constructor_id"].map(snapshot_constructors["constructor_coef_rapm"]).fillna(0.0)
    corrida["constructor_dnf_rate"] = corrida["constructor_id"].map(snapshot_constructors["constructor_dnf_rate"]).fillna(0.0)
    corrida["constructor_wins_total"] = corrida["constructor_id"].map(snapshot_constructors["constructor_wins_total"]).fillna(0.0)

    # features de circuito - lookup pelo número do round
    circ = circuito_features[circuito_features["round"] == round_num]
    for col in ["track_complexity", "altitude_m", "avg_pit_stops_circuit", "incident_rate_hist_norm"]:
        corrida[col] = circ.iloc[0][col] if (not circ.empty and col in circ.columns) else np.nan

    if circ.empty:
        log_lines.append(f"  circuito: round {round_num} sem histórico - features de circuito como NaN")

    # pneu de largada
    corrida["tire_compound_start"] = corrida["driver_id"].map(compound_series).fillna(2.0)

    # metadados da corrida
    corrida["season_factor"] = 2026
    corrida["safety_car_flag"] = safety_car
    corrida["season"] = 2026
    corrida["round"] = round_num
    corrida["race_name"] = race_name
    corrida["RaceID"] = corrida["driver_id"].astype(str) + "_2026_" + str(round_num)

    return corrida


def main():
    print("=" * 60)
    print("update_2026.py  (FastF1 + Jolpica)")
    print(f"Executado em: {datetime.now().isoformat(timespec='seconds')}")
    print("=" * 60)

    log_lines: list[str] = [
        "Relatório - update_2026.py (FastF1 + Jolpica)",
        f"Gerado em: {datetime.now().isoformat(timespec='seconds')}",
        "",
    ]

    # carrega o calendario 2026 via FastF1 (que por baixo usa a Jolpica)
    print("\n[1] Carregando calendário 2026 via FastF1/Jolpica...")
    try:
        schedule = fastf1.get_event_schedule(2026, include_testing=False)
    except Exception as exc:
        print(f"  [ERRO] Não foi possível carregar o calendário 2026: {exc}")
        return

    print(f"  {len(schedule)} eventos encontrados")
    log_lines.append(f"Calendário 2026: {len(schedule)} eventos")
    log_lines.append("")

    # carrega historico e monta snapshot do fim de 2025
    print("\n[2] Carregando features históricas do pipeline 2018-2025...")
    X, y = carregar_features_historicas()
    snapshot_drivers, snapshot_constructors = snapshot_features_fim_2025(X, y)
    circuito_features = carregar_features_circuito()
    print(f"  Snapshot drivers: {len(snapshot_drivers)} entidades")
    print(f"  Snapshot construtores: {len(snapshot_constructors)} entidades")

    print("\n[3] Processando corridas 2026...")
    frames: list[pd.DataFrame] = []

    for round_num in schedule["RoundNumber"].dropna().astype(int):
        try:
            event = schedule.get_event_by_round(round_num)
        except Exception as exc:
            log.error(f"  Não foi possível obter evento R{round_num}: {exc}")
            continue

        race_name = str(event["EventName"])

        df_corrida = processar_corrida(
            event=event,
            round_num=round_num,
            race_name=race_name,
            snapshot_drivers=snapshot_drivers,
            snapshot_constructors=snapshot_constructors,
            circuito_features=circuito_features,
            log_lines=log_lines,
        )

        if not df_corrida.empty:
            frames.append(df_corrida)

        time.sleep(1)  # respeita rate limit do Jolpica entre rounds

    if not frames:
        print("\n[ERRO] Nenhuma corrida 2026 processada - dados ainda não disponíveis.")
        log_lines.append("ERRO: nenhuma corrida processada.")
        OUTPUT_REPORT.write_text("\n".join(log_lines), encoding="utf-8")
        return

    df_final = pd.concat(frames, ignore_index=True)

    # se alguma feature tiver faltando, cria a coluna com NaN pra nao quebrar o schema
    missing = [f for f in FEATURES_FINAIS if f not in df_final.columns]
    if missing:
        log.warning(f"Features ausentes no output: {missing}")
        log_lines.append(f"Features ausentes: {missing}")
        for f in missing:
            df_final[f] = np.nan

    # valida cobertura de cada feature
    print("\n[4] Validação do dataset gerado...")
    log_lines.extend(["", "=== Cobertura das features ==="])
    for f in FEATURES_FINAIS:
        n_nulos = int(df_final[f].isna().sum())
        pct = n_nulos / len(df_final) * 100 if len(df_final) > 0 else 0
        status = "OK" if n_nulos == 0 else f"ATENÇÃO: {n_nulos} nulos ({pct:.1f}%)"
        log_lines.append(f"  {f}: {status}")
        print(f"  {f}: {status}")

    # monta as colunas na ordem certa e salva
    colunas_saida = KEY_COLS + [TARGET, "is_dnf", "safety_car_flag"] + FEATURES_FINAIS
    colunas_saida = [c for c in colunas_saida if c in df_final.columns]
    df_saida = (
        df_final[colunas_saida]
        .sort_values(["round", "finish_position"])
        .reset_index(drop=True)
    )

    df_saida.to_csv(OUTPUT_CSV, index=False)

    log_lines.extend([
        "",
        "=== Resumo final ===",
        f"Corridas processadas: {df_final['round'].nunique()}",
        f"Linhas totais: {len(df_saida)}",
        f"Pilotos únicos: {df_saida['driver_id'].nunique()}",
        f"NaN em qualifying_position: {int(df_saida['qualifying_position'].isna().sum())}",
        f"Output: {OUTPUT_CSV}",
    ])

    OUTPUT_REPORT.write_text("\n".join(log_lines), encoding="utf-8")

    print(f"\n[CONCLUÍDO]")
    print(f"  Corridas: {df_final['round'].nunique()}")
    print(f"  Linhas: {len(df_saida)}")
    print(f"  Output: {OUTPUT_CSV}")
    print(f"  Relatório: {OUTPUT_REPORT}")

    # avisa se o qualifying de alguma rodada ainda não veio
    if df_saida["qualifying_position"].isna().any():
        n_nan = int(df_saida["qualifying_position"].isna().sum())
        print(f"\n[ATENÇÃO] {n_nan} linhas sem qualifying_position - sessão de qualifying ainda não disponível.")


if __name__ == "__main__":
    main()
