from pathlib import Path

import pandas as pd


# preparação final da base antes de entrar no feature engineering
BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "data" / "processed"

INPUT_2018_2024 = PROCESSED_DIR / "dataset_pre_features_2018_2024.csv"
INPUT_2018_2025 = PROCESSED_DIR / "dataset_pre_features_2018_2025.csv"
DNF_CLASSIFICADO_2018_2025 = PROCESSED_DIR / "historico_dnf_classificado_2018_2025.csv"

OUTPUT_FE_2018_2024 = PROCESSED_DIR / "dataset_feature_engineering_ready_2018_2024.csv"
OUTPUT_FE_2018_2025 = PROCESSED_DIR / "dataset_feature_engineering_ready_2018_2025.csv"

TARGET_2018_2024 = PROCESSED_DIR / "target_finish_position_2018_2024.csv"
TARGET_2018_2025 = PROCESSED_DIR / "target_finish_position_2018_2025.csv"

OUTLIERS_REVISAO = PROCESSED_DIR / "outliers_revisao_2018_2025.csv"

TARGET = "finish_position"

# colunas que jamais podem entrar em X - finish_position fica na base só como target/historico
COLUNAS_PROIBIDAS_MODELO = [
    "finish_position",
    "points",
    "race_points",
    "fastest_lap_race",
    "previous_position",
]

# essas saem da base FE-ready pra evitar uso acidental; points é colinear demais com o resultado
COLUNAS_REMOVER_DA_BASE_FE = [
    "points",
    "race_points",
    "fastest_lap_race",
    "previous_position",
]

# tempos FastF1 são só auditoria, não entram como feature no modelo
OUTLIER_COLS_NAO_FEATURE = frozenset({
    "fastf1_avg_sector1",
    "fastf1_avg_sector2",
    "fastf1_avg_sector3",
    "fastf1_avg_lap_time",
    "fastf1_best_lap_time",
})

COLUNAS_CHAVE = [
    "season",
    "round",
    "RaceID",
    "driver_id",
    "constructor_id",
    "race_name",
    "circuit_id",
]

COLUNAS_OBRIGATORIAS = COLUNAS_CHAVE + [
    "grid_position",
    TARGET,
    "laps",
    "compound_ordinal",
    "safety_car_flag",
    "weather_impact_factor",
    "avg_pit_stops_circuit",
    "track_complexity",
    "qualifying_position",
    "grid_penalty",
]


def validar_base(df, nome):
    erros = []

    # checa se todas as colunas obrigatorias existem
    for coluna in COLUNAS_OBRIGATORIAS:
        if coluna not in df.columns:
            erros.append(f"{nome}: coluna obrigatoria ausente: {coluna}")

    if erros:
        return erros

    if df.empty:
        erros.append(f"{nome}: base vazia")

    duplicatas = int(df.duplicated("RaceID").sum())
    if duplicatas:
        erros.append(f"{nome}: {duplicatas} RaceIDs duplicados")

    # nulos nas colunas que a gente nao pode ter nulo
    nulos_obrigatorios = df[COLUNAS_OBRIGATORIAS].isna().sum()
    nulos_obrigatorios = nulos_obrigatorios[nulos_obrigatorios > 0]
    if not nulos_obrigatorios.empty:
        erros.append(f"{nome}: nulos em colunas obrigatorias: {nulos_obrigatorios.to_dict()}")

    if df["season"].min() < 2018:
        erros.append(f"{nome}: contem temporada anterior ao recorte oficial 2018")

    if (df["grid_position"] <= 0).any():
        erros.append(f"{nome}: grid_position <= 0 apos correcao de pit lane")

    if (df["finish_position"] <= 0).any():
        erros.append(f"{nome}: finish_position <= 0")

    if "points" in df.columns:
        # points pode existir na entrada da etapa, mas nao na saida FE-ready.
        pass

    return erros


def preparar_base(df):
    df = df.copy()

    # garante que essas colunas existam mesmo que a etapa anterior nao tenha criado
    if "tire_compound_start" not in df.columns:
        df["tire_compound_start"] = df["compound_ordinal"]

    if "season_factor" not in df.columns:
        df["season_factor"] = df["season"].astype(int)

    df = adicionar_avg_pitstops_causal(df)
    df = reconciliar_outliers_pos_contexto(df)
    df = reconciliar_outliers_nao_feature(df)
    df = enriquecer_track_complexity(df)

    # tira as colunas que nao devem circular na base FE-ready
    remover = [c for c in COLUNAS_REMOVER_DA_BASE_FE if c in df.columns]
    df = df.drop(columns=remover)

    return df


def reconciliar_outliers_pos_contexto(df):
    df = df.copy()

    colunas_necessarias = {"outlier_flag", "outlier_tipo", "safety_car_flag"}
    if not colunas_necessarias.issubset(df.columns):
        df["outlier_reclassificado_pos_contexto_flag"] = 0
        return df

    # outliers com safety car fazem sentido - não são anomalias, são contexto de pista
    mask_sc = (
        (df["outlier_flag"] == 1)
        & (df["outlier_tipo"] == "outlier_revisao")
        & (df["safety_car_flag"] == 1)
    )

    df.loc[mask_sc, "outlier_tipo"] = "outlier_legitimo"
    df.loc[mask_sc, "outlier_legitimo_flag"] = 1
    df.loc[mask_sc, "outlier_revisao_flag"] = 0

    df["outlier_reclassificado_pos_contexto_flag"] = mask_sc.astype(int)

    return df


def reconciliar_outliers_nao_feature(df):
    # se todas as colunas anomalas são de tempo FastF1 (que não entram no modelo),
    # o resultado da corrida é válido e promove pra outlier_legitimo
    df = df.copy()

    if "outlier_tipo" not in df.columns or "outlier_colunas" not in df.columns:
        df["outlier_reclassificado_nao_feature_flag"] = 0
        return df

    def _todas_nao_feature(cols_str):
        if not cols_str or str(cols_str).strip() == "":
            return False
        partes = [c.strip().rstrip(";") for c in str(cols_str).split(";") if c.strip()]
        return bool(partes) and all(c in OUTLIER_COLS_NAO_FEATURE for c in partes)

    mask = (
        (df["outlier_tipo"] == "outlier_revisao")
        & df["outlier_colunas"].apply(_todas_nao_feature)
    )

    df.loc[mask, "outlier_tipo"] = "outlier_legitimo"
    df.loc[mask, "outlier_legitimo_flag"] = 1
    df.loc[mask, "outlier_revisao_flag"] = 0
    df["outlier_reclassificado_nao_feature_flag"] = mask.astype(int)

    return df


def enriquecer_track_complexity(df):
    # recalcula track_complexity com taxa historica de SC/VSC por circuito
    # fórmula calibrada na validação temporal 2025
    df = df.copy()

    componentes = {"corners", "length_km", "altitude_m", "circuit_type", "safety_car_flag"}
    if not componentes.issubset(df.columns):
        df["track_complexity_static"] = df.get("track_complexity", 0.0)
        df["incident_rate_hist"] = float("nan")
        df["incident_rate_hist_norm"] = float("nan")
        return df

    # guarda a versão estática antiga pra comparar depois
    if "track_complexity" in df.columns:
        df["track_complexity_static"] = df["track_complexity"]

    treino = df[df["season"] <= 2024]

    # normaliza corners, length e altitude usando so os dados de treino como referencia
    norm_params = {}
    for col in ["corners", "length_km", "altitude_m"]:
        col_min = treino[col].min()
        col_max = treino[col].max()
        norm_params[col] = (col_min, col_max)
        df[f"_tc_{col}_norm"] = (
            (df[col] - col_min) / (col_max - col_min + 1e-9)
        ).clip(0, 1)

    # pega uma linha por corrida pra calcular a taxa de SC/VSC no nível de corrida
    race_sc = (
        df.groupby(["season", "round", "circuit_id"], as_index=False)["safety_car_flag"]
        .max()
        .sort_values(["season", "round"])
        .reset_index(drop=True)
    )

    # taxa histórica causal: só usa corridas anteriores no mesmo circuito
    race_sc["incident_rate_hist"] = (
        race_sc.groupby("circuit_id")["safety_car_flag"]
        .transform(lambda s: s.expanding().mean().shift(1))
    )

    # cold-start: usa a taxa global de 2018-2024 quando ainda nao tem historico do circuito
    global_rate = (
        treino.groupby(["season", "round"])["safety_car_flag"].max().mean()
    )
    race_sc["incident_rate_hist"] = race_sc["incident_rate_hist"].fillna(global_rate)

    # normaliza a taxa usando só o intervalo de treino
    rate_treino = race_sc[race_sc["season"] <= 2024]["incident_rate_hist"]
    rate_min = rate_treino.min()
    rate_max = rate_treino.max()
    race_sc["incident_rate_hist_norm"] = (
        (race_sc["incident_rate_hist"] - rate_min) / (rate_max - rate_min + 1e-9)
    ).clip(0, 1)

    causal = race_sc[
        ["season", "round", "circuit_id", "incident_rate_hist", "incident_rate_hist_norm"]
    ]
    df = df.merge(causal, on=["season", "round", "circuit_id"], how="left")
    df["incident_rate_hist_norm"] = df["incident_rate_hist_norm"].fillna(
        (global_rate - rate_min) / (rate_max - rate_min + 1e-9)
    ).clip(0, 1)

    # fórmula com 5 componentes calibrada pela validação 2025
    df["track_complexity"] = (
        0.358565 * df["_tc_corners_norm"]
        + 0.145285 * df["_tc_length_km_norm"]
        + 0.050026 * df["_tc_altitude_m_norm"]
        + 0.119041 * df["circuit_type"]
        + 0.327083 * df["incident_rate_hist_norm"]
    ).clip(0, 1)

    # joga fora as colunas temporárias de normalização
    df = df.drop(
        columns=["_tc_corners_norm", "_tc_length_km_norm", "_tc_altitude_m_norm"],
        errors="ignore",
    )

    return df


def adicionar_avg_pitstops_causal(df):
    # recalcula media de pit stops usando so corridas anteriores no circuito (sem leakage)
    df = df.copy()

    # guarda o valor antigo (global estático) só pra auditoria
    if "avg_pit_stops_circuit" in df.columns:
        df["avg_pit_stops_circuit_static_global"] = df["avg_pit_stops_circuit"]

    if "fastf1_pit_in_count" not in df.columns:
        return df

    # agrega no nível de corrida pra não depender de piloto
    race_level = (
        df.groupby(["season", "round", "circuit_id"], as_index=False)
        .agg(race_avg_pit_stops=("fastf1_pit_in_count", "mean"))
        .sort_values(["season", "round", "circuit_id"])
        .reset_index(drop=True)
    )

    # média histórica do circuito com shift(1) - só olha pra trás
    race_level["avg_pit_stops_circuit"] = (
        race_level.groupby("circuit_id")["race_avg_pit_stops"]
        .transform(lambda s: s.expanding().mean().shift(1))
    )

    race_level["avg_pit_stops_global_prior"] = (
        race_level["race_avg_pit_stops"].expanding().mean().shift(1)
    )

    race_level["avg_pit_stops_circuit_cold_start_flag"] = (
        race_level["avg_pit_stops_circuit"].isna()
    ).astype(int)

    # cold-start: usa media global anterior; se ainda nao tem nenhuma, coloca 0
    race_level["avg_pit_stops_circuit"] = (
        race_level["avg_pit_stops_circuit"]
        .fillna(race_level["avg_pit_stops_global_prior"])
        .fillna(0.0)
    )

    causal = race_level[
        [
            "season",
            "round",
            "circuit_id",
            "avg_pit_stops_circuit",
            "avg_pit_stops_circuit_cold_start_flag",
        ]
    ]

    # substitui a versao antiga pela causal no dataframe principal
    df = df.drop(
        columns=["avg_pit_stops_circuit", "avg_pit_stops_circuit_cold_start_flag"],
        errors="ignore",
    )
    df = df.merge(causal, on=["season", "round", "circuit_id"], how="left")
    return df


def resumir_outliers(df):
    if "outlier_flag" not in df.columns or "outlier_tipo" not in df.columns:
        return {
            "total_outliers": 0,
            "por_tipo": {},
            "reclassificados_pos_contexto": 0,
            "revisao_com_safety_car": 0,
        }

    outliers = df[df["outlier_flag"] == 1].copy()
    reclass_col = "outlier_reclassificado_pos_contexto_flag"

    if outliers.empty:
        return {
            "total_outliers": 0,
            "por_tipo": {},
            "reclassificados_pos_contexto": 0,
            "revisao_com_safety_car": 0,
        }

    # conta quantos outliers em revisao ainda tinham safety car
    revisao_com_sc = outliers[
        (outliers["outlier_tipo"] == "outlier_revisao")
        & (outliers.get("safety_car_flag", 0) == 1)
    ]

    return {
        "total_outliers": int(len(outliers)),
        "por_tipo": {
            str(k): int(v)
            for k, v in outliers["outlier_tipo"].value_counts().to_dict().items()
        },
        "reclassificados_pos_contexto": int(
            outliers[reclass_col].sum() if reclass_col in outliers.columns else 0
        ),
        "revisao_com_safety_car": int(len(revisao_com_sc)),
    }


def salvar_target(df, path):
    # salva só as colunas de identificação + finish_position
    target = df[COLUNAS_CHAVE + [TARGET]].copy()
    target.to_csv(path, index=False, encoding="utf-8-sig")


def salvar_outliers_revisao(df):
    if "outlier_tipo" not in df.columns:
        outliers = pd.DataFrame()
    else:
        outliers = df[df["outlier_tipo"] == "outlier_revisao"].copy()

    # pega as colunas mais uteis pra revisar manualmente
    colunas_preferidas = [
        "season",
        "round",
        "race_name",
        "driver_id",
        "constructor_id",
        "RaceID",
        "grid_position",
        "finish_position",
        "status",
        "fastf1_avg_lap_time",
        "fastf1_best_lap_time",
        "fastf1_avg_sector1",
        "fastf1_avg_sector2",
        "fastf1_avg_sector3",
        "outlier_colunas",
        "safety_car_flag",
        "corrida_chuva_flag",
        "outlier_tipo",
    ]
    colunas = [c for c in colunas_preferidas if c in outliers.columns]
    outliers = outliers[colunas].sort_values(["season", "round", "driver_id"])
    outliers.to_csv(OUTLIERS_REVISAO, index=False, encoding="utf-8-sig")
    return outliers


def main():
    print("Carregando bases pre-features...")
    df_2024_in = pd.read_csv(INPUT_2018_2024)
    df_2025_in = pd.read_csv(INPUT_2018_2025)

    # valida entrada antes de qualquer transformacao
    erros_entrada = []
    erros_entrada.extend(validar_base(df_2024_in, INPUT_2018_2024.name))
    erros_entrada.extend(validar_base(df_2025_in, INPUT_2018_2025.name))

    print("Preparando bases FE-ready...")
    df_2024 = preparar_base(df_2024_in)
    df_2025 = preparar_base(df_2025_in)

    # valida saída pra garantir que nenhuma transformação quebrou algo
    erros_saida = []
    erros_saida.extend(validar_base(df_2024, OUTPUT_FE_2018_2024.name))
    erros_saida.extend(validar_base(df_2025, OUTPUT_FE_2018_2025.name))

    if "points" in df_2024.columns or "points" in df_2025.columns:
        erros_saida.append("points ainda esta presente em base FE-ready")

    outliers = salvar_outliers_revisao(df_2025)
    salvar_target(df_2024, TARGET_2018_2024)
    salvar_target(df_2025, TARGET_2018_2025)

    df_2024.to_csv(OUTPUT_FE_2018_2024, index=False, encoding="utf-8-sig")
    df_2025.to_csv(OUTPUT_FE_2018_2025, index=False, encoding="utf-8-sig")

    print(f"FE-ready 2018-2024: {df_2024.shape}")
    print(f"FE-ready 2018-2025: {df_2025.shape}")
    print(f"Outliers em revisao: {len(outliers)}")

    if erros_entrada or erros_saida:
        print("\nErros encontrados:")
        for erro in erros_entrada + erros_saida:
            print(f"- {erro}")
        raise SystemExit(1)

    print("\nEtapa 09 finalizada com sucesso.")


if __name__ == "__main__":
    main()
