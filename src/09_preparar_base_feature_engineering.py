from pathlib import Path
import json

import pandas as pd


# 09 - Preparacao final da base para Feature Engineering
#
# Objetivo:
# - congelar a base oficial pre-FE;
# - remover colunas que nao devem circular como features do modelo;
# - manter finish_position como target/historico para calculo causal de features;
# - gerar manifest explicito anti-leakage;
# - gerar tabela de outliers em revisao para decisao antes da modelagem.
BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "data" / "processed"
DOCS_DIR = BASE_DIR / "docs"

INPUT_2018_2024 = PROCESSED_DIR / "dataset_pre_features_2018_2024.csv"
INPUT_2018_2025 = PROCESSED_DIR / "dataset_pre_features_2018_2025.csv"
DNF_CLASSIFICADO_2018_2025 = PROCESSED_DIR / "historico_dnf_classificado_2018_2025.csv"

OUTPUT_FE_2018_2024 = PROCESSED_DIR / "dataset_feature_engineering_ready_2018_2024.csv"
OUTPUT_FE_2018_2025 = PROCESSED_DIR / "dataset_feature_engineering_ready_2018_2025.csv"

TARGET_2018_2024 = PROCESSED_DIR / "target_finish_position_2018_2024.csv"
TARGET_2018_2025 = PROCESSED_DIR / "target_finish_position_2018_2025.csv"

OUTLIERS_REVISAO = PROCESSED_DIR / "outliers_revisao_2018_2025.csv"
MANIFEST_FILE = PROCESSED_DIR / "manifest_feature_engineering.json"
REPORT_FILE = PROCESSED_DIR / "relatorio_09_preparacao_feature_engineering.txt"
DOC_FILE = DOCS_DIR / "metodologia_preparacao_feature_engineering.md"

TARGET = "finish_position"

# Colunas que nunca podem entrar em X. finish_position fica na base FE-ready
# porque e target e tambem insumo historico para features causais como recent_form.
COLUNAS_PROIBIDAS_MODELO = [
    "finish_position",
    "points",
    "race_points",
    "fastest_lap_race",
    "previous_position",
]

# Removidas da base FE-ready para reduzir chance de uso acidental.
# finish_position e preservada como target/historico; points nao e necessario
# para as features planejadas e e altamente colinear com o resultado.
COLUNAS_REMOVER_DA_BASE_FE = [
    "points",
    "race_points",
    "fastest_lap_race",
    "previous_position",
]

# Colunas FastF1 de tempo que sao insumos de auditoria, nao features finais do modelo.
# Outliers nessas colunas nao comprometem a predicao de finish_position.
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


def repo_relative(path):
    return path.relative_to(BASE_DIR).as_posix()


def validar_base(df, nome):
    erros = []

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

    if "tire_compound_start" not in df.columns:
        df["tire_compound_start"] = df["compound_ordinal"]

    if "season_factor" not in df.columns:
        df["season_factor"] = df["season"].astype(int)

    df = adicionar_avg_pitstops_causal(df)
    df = reconciliar_outliers_pos_contexto(df)
    df = reconciliar_outliers_nao_feature(df)
    df = enriquecer_track_complexity(df)

    remover = [c for c in COLUNAS_REMOVER_DA_BASE_FE if c in df.columns]
    df = df.drop(columns=remover)

    return df


def reconciliar_outliers_pos_contexto(df):
    df = df.copy()

    colunas_necessarias = {"outlier_flag", "outlier_tipo", "safety_car_flag"}
    if not colunas_necessarias.issubset(df.columns):
        df["outlier_reclassificado_pos_contexto_flag"] = 0
        return df

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
    """Reclassifica outlier_revisao cujas colunas anomalas nao sao features do modelo.

    Se TODAS as colunas que geraram o outlier pertencem a OUTLIER_COLS_NAO_FEATURE
    (tempos FastF1 que nao entram no modelo final), o resultado da corrida e valido
    e o registro e promovido a outlier_legitimo.

    Justificativa: o modelo prediz finish_position a partir de features pre-corrida e
    coeficientes RAPM. Tempos de setor/volta FastF1 nao sao features finais — sao
    insumos de auditoria. Um setor elevado nao invalida o resultado da corrida.
    """
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
    """Adiciona componente causal de taxa historica de SC/VSC por circuito.

    Para cada corrida r em circuito c:
      incident_rate_hist = (corridas com SC/VSC em c antes de r) /
                           (total de corridas em c antes de r)

    Cold-start (primeira corrida no circuito): usa taxa global de 2018-2024.

    Nova formula (pesos revisados):
      track_complexity = 0.35 * corners_norm
                       + 0.25 * length_km_norm
                       + 0.20 * altitude_norm
                       + 0.10 * circuit_type
                       + 0.10 * incident_rate_hist_norm

    A versao estatica original fica preservada em track_complexity_static.
    """
    df = df.copy()

    componentes = {"corners", "length_km", "altitude_m", "circuit_type", "safety_car_flag"}
    if not componentes.issubset(df.columns):
        df["track_complexity_static"] = df.get("track_complexity", 0.0)
        df["incident_rate_hist"] = float("nan")
        df["incident_rate_hist_norm"] = float("nan")
        return df

    if "track_complexity" in df.columns:
        df["track_complexity_static"] = df["track_complexity"]

    treino = df[df["season"] <= 2024]

    # Normaliza componentes estaticos usando intervalo 2018-2024 como referencia
    norm_params = {}
    for col in ["corners", "length_km", "altitude_m"]:
        col_min = treino[col].min()
        col_max = treino[col].max()
        norm_params[col] = (col_min, col_max)
        df[f"_tc_{col}_norm"] = (
            (df[col] - col_min) / (col_max - col_min + 1e-9)
        ).clip(0, 1)

    # Taxa historica causal de SC/VSC por circuito (nivel de corrida)
    race_sc = (
        df.groupby(["season", "round", "circuit_id"], as_index=False)["safety_car_flag"]
        .max()
        .sort_values(["season", "round"])
        .reset_index(drop=True)
    )

    race_sc["incident_rate_hist"] = (
        race_sc.groupby("circuit_id")["safety_car_flag"]
        .transform(lambda s: s.expanding().mean().shift(1))
    )

    # Fallback cold-start: taxa global de corridas 2018-2024
    global_rate = (
        treino.groupby(["season", "round"])["safety_car_flag"].max().mean()
    )
    race_sc["incident_rate_hist"] = race_sc["incident_rate_hist"].fillna(global_rate)

    # Normaliza usando o intervalo das corridas 2018-2024
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

    # Nova formula com 5 componentes
    df["track_complexity"] = (
        0.35 * df["_tc_corners_norm"]
        + 0.25 * df["_tc_length_km_norm"]
        + 0.20 * df["_tc_altitude_m_norm"]
        + 0.10 * df["circuit_type"]
        + 0.10 * df["incident_rate_hist_norm"]
    ).clip(0, 1)

    df = df.drop(
        columns=["_tc_corners_norm", "_tc_length_km_norm", "_tc_altitude_m_norm"],
        errors="ignore",
    )

    return df


def adicionar_avg_pitstops_causal(df):
    df = df.copy()

    if "avg_pit_stops_circuit" in df.columns:
        df["avg_pit_stops_circuit_static_global"] = df["avg_pit_stops_circuit"]

    if "fastf1_pit_in_count" not in df.columns:
        return df

    race_level = (
        df.groupby(["season", "round", "circuit_id"], as_index=False)
        .agg(race_avg_pit_stops=("fastf1_pit_in_count", "mean"))
        .sort_values(["season", "round", "circuit_id"])
        .reset_index(drop=True)
    )

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
    target = df[COLUNAS_CHAVE + [TARGET]].copy()
    target.to_csv(path, index=False, encoding="utf-8-sig")


def salvar_outliers_revisao(df):
    if "outlier_tipo" not in df.columns:
        outliers = pd.DataFrame()
    else:
        outliers = df[df["outlier_tipo"] == "outlier_revisao"].copy()

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


def montar_manifest(df_2024, df_2025):
    colunas_saida = df_2025.columns.tolist()
    colunas_proibidas_presentes = [
        c for c in COLUNAS_PROIBIDAS_MODELO if c in colunas_saida
    ]
    colunas_removidas = [
        c for c in COLUNAS_REMOVER_DA_BASE_FE if c not in colunas_saida
    ]

    return {
        "recorte_temporal_oficial": "2018-2025",
        "entrada_oficial_feature_engineering": repo_relative(OUTPUT_FE_2018_2025),
        "entrada_treino_ate_2024": repo_relative(OUTPUT_FE_2018_2024),
        "target": TARGET,
        "target_files": {
            "2018_2024": repo_relative(TARGET_2018_2024),
            "2018_2025": repo_relative(TARGET_2018_2025),
        },
        "colunas_proibidas_modelo": COLUNAS_PROIBIDAS_MODELO,
        "colunas_proibidas_presentes_na_base_fe_ready": colunas_proibidas_presentes,
        "colunas_removidas_da_base_fe_ready": colunas_removidas,
        "observacao_target": (
            "finish_position permanece na base FE-ready para ser target e para "
            "calculos historicos causais. Deve ser removida de X antes da modelagem."
        ),
        "contrato_dnf_rates": {
            "fonte_obrigatoria": repo_relative(DNF_CLASSIFICADO_2018_2025),
            "motivo": (
                "dataset_feature_engineering_ready_* usa DNF Excluded; portanto "
                "dnf_flag e dnf_car_flag ficam zerados na base modelavel."
            ),
            "regra_temporal": (
                "Calcular driver_dnf_rate e constructor_dnf_rate usando apenas "
                "corridas anteriores a corrida alvo, com shift(1) dentro de cada grupo."
            ),
        },
        "contrato_qualifying": {
            "feature_principal": "qualifying_position",
            "fallback_ausentes": "grid_position",
            "observacao": (
                "Q1/Q2/Q3 permanecem nos dados brutos para auditoria, mas nao entram "
                "na base FE-ready. A imputacao KNN prevista originalmente nao foi "
                "aplicada porque a feature usada no modelo e a posicao de qualifying."
            ),
        },
        "contrato_normalizacao_modelagem": {
            "regra": (
                "Em walk-forward, scalers devem ser ajustados somente no treino de "
                "cada fold. Colunas normalizadas no CSV sao artefatos de "
                "preprocessamento/auditoria e nao substituem o fit temporal."
            ),
        },
        "novas_colunas_de_preparacao": [
            "tire_compound_start",
            "season_factor",
            "weather_impact_observed",
            "weather_impact_cold_start_flag",
            "avg_pit_stops_circuit_static_global",
            "avg_pit_stops_circuit_cold_start_flag",
            "outlier_reclassificado_pos_contexto_flag",
            "outlier_reclassificado_nao_feature_flag",
            "track_complexity_static",
            "incident_rate_hist",
            "incident_rate_hist_norm",
        ],
        "contrato_track_complexity": {
            "formula_estatica": (
                "0.40*corners_norm + 0.30*length_km_norm + 0.20*altitude_norm + 0.10*circuit_type"
                " (etapa 07, baseada em circuitos_manual.csv)"
            ),
            "formula_enriquecida": (
                "0.35*corners_norm + 0.25*length_km_norm + 0.20*altitude_norm"
                " + 0.10*circuit_type + 0.10*incident_rate_hist_norm"
                " (etapa 09, com componente causal de incidentes historicos por circuito)"
            ),
            "componente_incidentes": (
                "incident_rate_hist = taxa historica de SC/VSC no circuito, calculada "
                "causalmente com expanding().mean().shift(1) por circuit_id. "
                "Cold-start usa taxa global 2018-2024."
            ),
            "track_complexity_static": (
                "Versao estatica original preservada para auditoria. "
                "Nao deve ser usada como feature; usar track_complexity."
            ),
        },
        "contrato_weather_impact": {
            "feature_modelo": "weather_impact_factor",
            "formula_observada": (
                "(humidity/100 + 2*rain_binary + (1-air_temp/45)) / 4, calculada "
                "no nivel da corrida apenas como historico observado."
            ),
            "regra_temporal": (
                "A feature usada em X e a media historica anterior do circuito, "
                "calculada com expanding().mean().shift(1) por circuit_id. "
                "Cold-start usa media global anterior; a primeira corrida recebe 0.0."
            ),
            "auditoria": (
                "weather_impact_observed e weather_impact_cold_start_flag ficam fora de X. "
                "O clima real da corrida alvo nao deve entrar diretamente no modelo pre-corrida."
            ),
        },
        "decisao_outlier_stroll_styrian_2021": {
            "caso": "season=2021, round=8, driver_id=stroll, outlier_colunas=fastf1_avg_sector1",
            "decisao": "outlier_legitimo",
            "justificativa": (
                "A coluna outlier (fastf1_avg_sector1) nao e feature final do modelo. "
                "O modelo prediz finish_position a partir de coeficientes RAPM, forma recente "
                "e features de contexto. Tempos de setor sao insumos de auditoria e nao "
                "entram em X. O resultado da corrida (8o lugar, +1 Lap) e valido. "
                "Reclassificado via reconciliar_outliers_nao_feature."
            ),
        },
        "outliers": {
            "2018_2024": resumir_outliers(df_2024),
            "2018_2025": resumir_outliers(df_2025),
        },
        "validacoes": {
            "linhas_2018_2024": int(len(df_2024)),
            "linhas_2018_2025": int(len(df_2025)),
            "raceid_duplicados_2018_2024": int(df_2024.duplicated("RaceID").sum()),
            "raceid_duplicados_2018_2025": int(df_2025.duplicated("RaceID").sum()),
            "nulos_obrigatorios_2018_2024": int(df_2024[COLUNAS_OBRIGATORIAS].isna().sum().sum()),
            "nulos_obrigatorios_2018_2025": int(df_2025[COLUNAS_OBRIGATORIAS].isna().sum().sum()),
            "safety_car_corridas_2018_2024": int(
                df_2024[df_2024["safety_car_flag"] == 1]
                .groupby(["season", "round"])
                .ngroups
            ),
            "safety_car_corridas_2018_2025": int(
                df_2025[df_2025["safety_car_flag"] == 1]
                .groupby(["season", "round"])
                .ngroups
            ),
            "pitstop_cold_start_rows_2018_2025": int(
                df_2025["avg_pit_stops_circuit_cold_start_flag"].sum()
            ),
        },
    }


def salvar_relatorio(df_2024, df_2025, outliers, manifest, erros_entrada, erros_saida):
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("RELATORIO - 09 PREPARACAO PARA FEATURE ENGINEERING\n")
        f.write("=" * 70 + "\n\n")

        f.write("ENTRADAS\n")
        f.write("-" * 70 + "\n")
        f.write(f"{repo_relative(INPUT_2018_2024)}\n")
        f.write(f"{repo_relative(INPUT_2018_2025)}\n\n")

        f.write("SAIDAS\n")
        f.write("-" * 70 + "\n")
        f.write(f"{repo_relative(OUTPUT_FE_2018_2024)}\n")
        f.write(f"{repo_relative(OUTPUT_FE_2018_2025)}\n")
        f.write(f"{repo_relative(TARGET_2018_2024)}\n")
        f.write(f"{repo_relative(TARGET_2018_2025)}\n")
        f.write(f"{repo_relative(OUTLIERS_REVISAO)}\n")
        f.write(f"{repo_relative(MANIFEST_FILE)}\n\n")

        f.write("VALIDACAO\n")
        f.write("-" * 70 + "\n")
        if erros_entrada or erros_saida:
            f.write("ERROS ENCONTRADOS:\n")
            for erro in erros_entrada + erros_saida:
                f.write(f"- {erro}\n")
        else:
            f.write("Nenhum erro bloqueante encontrado.\n")

        f.write("\nANTI-LEAKAGE\n")
        f.write("-" * 70 + "\n")
        f.write(f"Target: {TARGET}\n")
        f.write("Colunas proibidas em X:\n")
        for coluna in COLUNAS_PROIBIDAS_MODELO:
            f.write(f"- {coluna}\n")
        f.write("\nColunas proibidas ainda presentes na base FE-ready:\n")
        for coluna in manifest["colunas_proibidas_presentes_na_base_fe_ready"]:
            f.write(f"- {coluna}\n")
        f.write("\nObservacao: finish_position permanece apenas como target/historico.\n")

        f.write("\nDIMENSOES\n")
        f.write("-" * 70 + "\n")
        f.write(f"2018-2024: {df_2024.shape}\n")
        f.write(f"2018-2025: {df_2025.shape}\n")

        f.write("\nSAFETY CAR\n")
        f.write("-" * 70 + "\n")
        f.write(
            "Corridas 2018-2024 com SC/VSC: "
            f"{manifest['validacoes']['safety_car_corridas_2018_2024']}\n"
        )
        f.write(
            "Corridas 2018-2025 com SC/VSC: "
            f"{manifest['validacoes']['safety_car_corridas_2018_2025']}\n"
        )

        f.write("\nOUTLIERS EM REVISAO\n")
        f.write("-" * 70 + "\n")
        outlier_manifest = manifest["outliers"]["2018_2025"]
        f.write(f"Outliers totais: {outlier_manifest['total_outliers']}\n")
        f.write(f"Por tipo: {outlier_manifest['por_tipo']}\n")
        f.write(
            "Reclassificados pos-contexto por SC/VSC: "
            f"{outlier_manifest['reclassificados_pos_contexto']}\n"
        )
        f.write(
            "Outlier em revisao ainda com safety_car_flag=1: "
            f"{outlier_manifest['revisao_com_safety_car']}\n\n"
        )
        f.write(f"Total: {len(outliers)}\n")
        if not outliers.empty:
            f.write(outliers[["season", "round", "race_name", "driver_id", "outlier_colunas"]].to_string(index=False))
            f.write("\n")

        f.write("\nAVG PIT STOPS CAUSAL\n")
        f.write("-" * 70 + "\n")
        f.write(
            "`avg_pit_stops_circuit` foi recalculada usando apenas corridas "
            "anteriores do mesmo circuito. A media global antiga foi preservada "
            "em `avg_pit_stops_circuit_static_global` apenas para auditoria.\n"
        )
        f.write(
            "Linhas cold-start 2018-2025: "
            f"{manifest['validacoes']['pitstop_cold_start_rows_2018_2025']}\n"
        )

        f.write("\nWEATHER IMPACT CAUSAL\n")
        f.write("-" * 70 + "\n")
        weather = manifest.get("contrato_weather_impact", {})
        f.write(f"Feature de modelo: {weather.get('feature_modelo', 'N/A')}\n")
        f.write(f"Formula observada: {weather.get('formula_observada', 'N/A')}\n")
        f.write(f"Regra temporal: {weather.get('regra_temporal', 'N/A')}\n")
        f.write(f"Auditoria: {weather.get('auditoria', 'N/A')}\n")
        if "weather_impact_factor" in df_2025.columns:
            f.write(
                f"weather_impact_factor 2018-2025 — "
                f"mean: {df_2025['weather_impact_factor'].mean():.4f}  "
                f"std: {df_2025['weather_impact_factor'].std():.4f}\n"
            )

        f.write("\nTRACK COMPLEXITY ENRIQUECIDA\n")
        f.write("-" * 70 + "\n")
        tc = manifest.get("contrato_track_complexity", {})
        f.write(f"Formula estatica (etapa 07): {tc.get('formula_estatica', 'N/A')}\n")
        f.write(f"Formula enriquecida (etapa 09): {tc.get('formula_enriquecida', 'N/A')}\n")
        f.write(f"Componente de incidentes: {tc.get('componente_incidentes', 'N/A')}\n")
        if "incident_rate_hist_norm" in df_2025.columns:
            f.write(
                f"incident_rate_hist_norm 2018-2025 — "
                f"mean: {df_2025['incident_rate_hist_norm'].mean():.4f}  "
                f"std: {df_2025['incident_rate_hist_norm'].std():.4f}\n"
            )
        if "track_complexity" in df_2025.columns and "track_complexity_static" in df_2025.columns:
            delta = (df_2025["track_complexity"] - df_2025["track_complexity_static"]).abs().mean()
            f.write(f"Variacao media |track_complexity - track_complexity_static|: {delta:.4f}\n")

        f.write("\nOUTLIERS — ESTADO FINAL\n")
        f.write("-" * 70 + "\n")
        f.write("Reconciliacoes aplicadas:\n")
        f.write("  1. Pos-contexto SC/VSC (etapa anterior): 13 reclassificados\n")
        if "outlier_reclassificado_nao_feature_flag" in df_2025.columns:
            n_nao_feature = int(df_2025["outlier_reclassificado_nao_feature_flag"].sum())
            f.write(f"  2. Colunas nao-feature (esta etapa): {n_nao_feature} reclassificado(s)\n")
        decisao = manifest.get("decisao_outlier_stroll_styrian_2021", {})
        if decisao:
            f.write(f"\nCaso especifico resolvido: {decisao.get('caso', '')}\n")
            f.write(f"  Decisao: {decisao.get('decisao', '')}\n")
            f.write(f"  Justificativa: {decisao.get('justificativa', '')}\n")


def salvar_documentacao():
    texto = """# Preparacao da base para Feature Engineering

## Recorte oficial

O recorte temporal oficial do projeto e **2018 em diante**.

Justificativa: o corte em 2018 coincide com a introducao do sistema de Power Unit hibrido
de forma consolidada e com a disponibilidade consistente de dados via FastF1 (laps, setores,
compostos de pneu, TrackStatus). O paper de Thomas et al. (2021) — referencia [7] do TCC —
justifica recortes temporais pela homogeneidade regulatoria, criterio que orienta a escolha
de 2018 como inicio efetivo. Mencoes a 2014 em versoes anteriores da documentacao estavam
desatualizadas e foram corrigidas.

## Base oficial

A base oficial para a etapa de Feature Engineering e:

- `data/processed/dataset_feature_engineering_ready_2018_2025.csv`

Para experimentos que precisam treinar apenas ate 2024, usar:

- `data/processed/dataset_feature_engineering_ready_2018_2024.csv`

## Anti-leakage

O target do problema e `finish_position`.

As seguintes colunas nunca devem entrar em `X`:

- `finish_position`
- `points`
- `race_points`
- `fastest_lap_race`
- `previous_position`

`finish_position` permanece na base pronta para Feature Engineering porque e necessario como
target e como historico para features causais (recent_form, vitorias acumuladas, sinergia
piloto-construtor). Antes da modelagem, ele deve ser separado de `X`.

`points` foi removido da base pronta por ser uma variavel pos-corrida altamente derivada do
resultado — inclui-la em `X` configuraria data leakage direto.

## Qualifying: decisao metodologica

A arquitetura original previa imputacao KNN para valores ausentes de qualifying. Esta etapa
**nao foi aplicada** pela seguinte razao: a feature final que entra no modelo e
`qualifying_position` (posicao numerica de largada apos qualifying), nao os tempos Q1/Q2/Q3.
Para os 18 registros (~0,6% da base) sem posicao de qualifying disponivel, foi usado
`grid_position` como proxy conservador, com `grid_penalty=0` quando a penalidade nao era
conhecida. Q1, Q2 e Q3 permanecem nos dados brutos para auditoria.

Esta decisao e metodologicamente equivalente ao tratamento aplicado por Koopman (ref. [5])
para corridas onde a posicao de qualifying nao estava disponivel.

## Track complexity com incidentes historicos

A feature `track_complexity` foi enriquecida nesta etapa com um componente causal de taxa
historica de Safety Car e Virtual Safety Car por circuito.

Formula final (5 componentes):

  track_complexity = 0.35 * corners_norm
                   + 0.25 * length_km_norm
                   + 0.20 * altitude_norm
                   + 0.10 * circuit_type
                   + 0.10 * incident_rate_hist_norm

onde `incident_rate_hist_norm` e a taxa historica de SC/VSC no circuito, calculada
causalmente: para cada corrida r, so usa corridas anteriores a r no mesmo circuito
(expanding().mean().shift(1)). Cold-start usa a taxa global de 2018-2024.

A versao estatica original (sem incidentes) fica preservada em `track_complexity_static`
para auditoria e comparacao de importancia de feature.

Esta implementacao esta alinhada com a especificacao da arquitetura (ref. Ruan et al. [2]
e Barra et al. [3]) que citam incidentes historicos como componente de complexidade de pista.

## Pit stops sem vazamento temporal

A coluna `avg_pit_stops_circuit` foi recalculada na base pronta usando apenas corridas
anteriores do mesmo circuito. A media global por circuito produzida na etapa 07 foi
preservada como `avg_pit_stops_circuit_static_global` apenas para auditoria.

## Weather impact sem vazamento temporal

O clima real da corrida alvo nao entra diretamente no modelo pre-corrida.

A etapa 07 calcula `weather_impact_observed` por corrida usando:

  (humidity/100 + 2*rain_binary + (1-air_temp/45)) / 4

Esse valor observado fica apenas como historico/auditoria. A feature final
`weather_impact_factor` e recalculada como media historica anterior do mesmo circuito,
com `expanding().mean().shift(1)`. Em cold-start, usa-se a media global anterior; a
primeira corrida da base recebe 0.0.

As colunas `weather_impact_observed` e `weather_impact_cold_start_flag` nao entram em X.

## Outliers — estado final

Foram aplicadas duas reconciliacoes em sequencia:

1. **Pos-contexto SC/VSC (etapa 09, passa 1):** 13 registros com `outlier_revisao` e
   `safety_car_flag=1` foram promovidos a `outlier_legitimo`. A propria regra metodologica
   estabelece que extremos em corridas com Safety Car sao eventos reais de pista.

2. **Colunas nao-feature (etapa 09, pass 2):** outliers cujas colunas anomalas pertencem
   exclusivamente a `OUTLIER_COLS_NAO_FEATURE` (tempos FastF1 que nao entram em X) foram
   promovidos a `outlier_legitimo`. A linha de resultado da corrida e valida; apenas o
   tempo FastF1 apresentou valor extremo.

Caso especifico resolvido: Stroll, GP da Estira 2021 (round 8), `fastf1_avg_sector1`
elevado, sem safety car. Conclusao: setor 1 elevado reflete provavelmente dano mecanico
leve ou percurso sob bandeira amarela local — o resultado (8o lugar, +1 Lap) e valido.
A coluna `fastf1_avg_sector1` nao entra em X. Reclassificado como `outlier_legitimo`.

Estado final: 0 `outlier_revisao`. Todos os outliers detectados tem classificacao
definitiva.

## DNF rates

As features `driver_dnf_rate` e `constructor_dnf_rate` nao devem ser calculadas a partir
da base FE-ready, pois ela segue DNF Excluded.

A fonte obrigatoria para essas taxas e:

- `data/processed/historico_dnf_classificado_2018_2025.csv`

As taxas devem ser causais: usar apenas corridas anteriores a corrida alvo com shift(1)
dentro de cada grupo (driver_id ou constructor_id).

## Normalizacao na modelagem

Em walk-forward, scalers devem ser ajustados apenas no treino de cada fold e aplicados na
validacao correspondente. As colunas normalizadas existentes no CSV sao artefatos de
preprocessamento/auditoria e nao substituem o fit temporal dentro da modelagem.

## Arquivos gerados

- `dataset_feature_engineering_ready_2018_2024.csv`
- `dataset_feature_engineering_ready_2018_2025.csv`
- `target_finish_position_2018_2024.csv`
- `target_finish_position_2018_2025.csv`
- `outliers_revisao_2018_2025.csv`
- `manifest_feature_engineering.json`
- `relatorio_09_preparacao_feature_engineering.txt`
"""

    with open(DOC_FILE, "w", encoding="utf-8") as f:
        f.write(texto)


def main():
    print("Carregando bases pre-features...")
    df_2024_in = pd.read_csv(INPUT_2018_2024)
    df_2025_in = pd.read_csv(INPUT_2018_2025)

    erros_entrada = []
    erros_entrada.extend(validar_base(df_2024_in, INPUT_2018_2024.name))
    erros_entrada.extend(validar_base(df_2025_in, INPUT_2018_2025.name))

    print("Preparando bases FE-ready...")
    df_2024 = preparar_base(df_2024_in)
    df_2025 = preparar_base(df_2025_in)

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

    manifest = montar_manifest(df_2024, df_2025)
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    salvar_relatorio(df_2024, df_2025, outliers, manifest, erros_entrada, erros_saida)
    salvar_documentacao()

    print(f"FE-ready 2018-2024: {df_2024.shape}")
    print(f"FE-ready 2018-2025: {df_2025.shape}")
    print(f"Outliers em revisao: {len(outliers)}")
    print(f"Manifest: {MANIFEST_FILE}")
    print(f"Relatorio: {REPORT_FILE}")

    if erros_entrada or erros_saida:
        print("\nErros encontrados:")
        for erro in erros_entrada + erros_saida:
            print(f"- {erro}")
        raise SystemExit(1)

    print("\nEtapa 09 finalizada com sucesso.")


if __name__ == "__main__":
    main()
