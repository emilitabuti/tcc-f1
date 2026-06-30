from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 12 - Análise de correlação e tratamento final de NaN

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports" / "correlacao_features"

INPUT_FILE = PROCESSED_DIR / "dataset_features_final_2018_2025.csv"

OUTPUT_DATASET = PROCESSED_DIR / "dataset_features_final_2018_2025_sem_nan.csv"
OUTPUT_CORRELATION_MATRIX = REPORTS_DIR / "matriz_correlacao_features.csv"
OUTPUT_CORRELATION_MATRIX_PNG = REPORTS_DIR / "correlation_matrix_features.png"
OUTPUT_TARGET_CORRELATION = REPORTS_DIR / "correlation_with_target.csv"
OUTPUT_HIGH_CORRELATION = REPORTS_DIR / "pares_correlacao_alta_maior_085.csv"


TARGET_COLUMNS = [
    "finish_position",
]

# features canônicas que entram na análise de correlação
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


# quais colunas preenchem NaN com zero (flags e coeficientes sem histórico)
FEATURES_NAN_ZERO = [
    "driver_coef_rapm",
    "constructor_coef_rapm",
    "recent_form_5",
    "recent_form_3",
    "recent_form_cold_start_flag",
    "driver_experience",
    "driver_win_flag",
    "driver_wins_total",
    "constructor_wins_total",
    "driver_dnf_rate",
    "constructor_dnf_rate",
    "driver_constructor_synergy",
    "avg_pit_stops_circuit_cold_start_flag",
    "safety_car_flag",
    "corrida_chuva_flag",
    "wet_compound_flag",
    "outlier_flag",
    "outlier_legitimo_flag",
    "outlier_revisao_flag",
    "outlier_espurio_flag",
    "outlier_reclassificado_pos_contexto_flag",
    "outlier_reclassificado_nao_feature_flag",
    "status_falha_mecanica_flag",
]


# quais colunas preenchem NaN com mediana
FEATURES_NAN_MEDIAN = [
    "grid_position",
    "laps",
    "fastf1_laps_count",
    "fastf1_avg_lap_time",
    "fastf1_best_lap_time",
    "fastf1_avg_sector1",
    "fastf1_avg_sector2",
    "fastf1_avg_sector3",
    "fastf1_max_tyre_life",
    "fastf1_stints_count",
    "fastf1_pit_in_count",
    "fastf1_pit_out_count",
    "qualifying_position",
    "grid_penalty",
    "altitude_m",
    "corners",
    "length_km",
    "track_complexity",
    "weather_impact_factor",
    "weather_impact_cold_start_flag",
    "avg_pit_stops_circuit",
    "track_complexity_static",
    "incident_rate_hist",
    "incident_rate_hist_norm",
]


def carregar_dataset():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    # só pra garantir que não tem RaceID duplicado antes de continuar
    if "RaceID" in df.columns:
        duplicados = df["RaceID"].duplicated().sum()
        if duplicados > 0:
            raise RuntimeError(f"Foram encontrados {duplicados} RaceID duplicados.")

    return df


def validar_features_finais(df):
    faltantes = [col for col in FEATURES_FINAIS + TARGET_COLUMNS if col not in df.columns]

    if faltantes:
        raise ValueError(f"Colunas finais ausentes no dataset: {faltantes}")


def tratar_nan(df):
    df = df.copy()

    # conta quantos NaN existem antes de tratar
    resumo_nan_antes = df.isna().sum()
    resumo_nan_antes = resumo_nan_antes[resumo_nan_antes > 0].sort_values(ascending=False)

    # flags e coeficientes sem histórico viram 0
    for col in FEATURES_NAN_ZERO:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # variáveis contínuas recebem mediana
    for col in FEATURES_NAN_MEDIAN:
        if col in df.columns:
            mediana = df[col].median()
            df[col] = df[col].fillna(mediana)

    # Para colunas numéricas restantes, usa mediana como fallback.
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols:
        if df[col].isna().sum() > 0:
            mediana = df[col].median()
            df[col] = df[col].fillna(mediana)

    # Para colunas de texto restantes, usa "desconhecido".
    object_cols = df.select_dtypes(include=["object"]).columns.tolist()

    for col in object_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna("desconhecido")

    # confere o que sobrou depois do tratamento
    resumo_nan_depois = df.isna().sum()
    resumo_nan_depois = resumo_nan_depois[resumo_nan_depois > 0].sort_values(ascending=False)

    return df, resumo_nan_antes, resumo_nan_depois


def selecionar_features_numericas(df):
    return FEATURES_FINAIS.copy()


def gerar_matriz_correlacao(df, features):
    # correlação de Pearson entre as features finais
    corr = df[features].corr(method="pearson")

    return corr


def gerar_correlacao_target(df, features):
    # correlação de cada feature com o target, ordenada por valor absoluto
    corr_target = (
        df[features + TARGET_COLUMNS]
        .corr(method="pearson")[TARGET_COLUMNS[0]]
        .drop(TARGET_COLUMNS[0])
        .sort_values(key=lambda s: s.abs(), ascending=False)
        .reset_index()
    )
    corr_target.columns = ["feature", "correlacao_com_finish_position"]

    return corr_target


def salvar_heatmap_correlacao(corr):
    plt.figure(figsize=(14, 11))
    sns.heatmap(
        corr,
        cmap="vlag",
        center=0,
        square=True,
        linewidths=0.4,
        cbar_kws={"shrink": 0.75},
    )
    plt.title("Matriz de correlação - features finais")
    plt.tight_layout()
    plt.savefig(OUTPUT_CORRELATION_MATRIX_PNG, dpi=180)
    plt.close()


def identificar_correlacoes_altas(corr, limite=0.85):
    pares = []

    colunas = corr.columns.tolist()

    # varre o triângulo superior da matriz pra não repetir pares
    for i in range(len(colunas)):
        for j in range(i + 1, len(colunas)):
            col_a = colunas[i]
            col_b = colunas[j]
            valor = corr.loc[col_a, col_b]

            if pd.notna(valor) and abs(valor) > limite:
                pares.append(
                    {
                        "feature_1": col_a,
                        "feature_2": col_b,
                        "correlacao": valor,
                        "correlacao_abs": abs(valor),
                    }
                )

    pares_df = pd.DataFrame(pares)

    if not pares_df.empty:
        pares_df = pares_df.sort_values(
            "correlacao_abs",
            ascending=False,
        ).reset_index(drop=True)

    return pares_df


def sugerir_decisao(feature_1, feature_2):
    # features que têm prioridade metodológica e devem ser mantidas
    regras_manter = [
        "driver_coef_rapm",
        "constructor_coef_rapm",
        "recent_form_5",
        "driver_experience",
        "driver_wins_total",
        "constructor_wins_total",
        "driver_dnf_rate",
        "constructor_dnf_rate",
        "driver_constructor_synergy",
        "track_complexity",
        "weather_impact_factor",
        "avg_pit_stops_circuit",
        "laps_minmax",
        "incident_rate_hist_norm",
    ]

    # Se uma é versão normalizada/zscore e a outra é original,
    # sugere manter a versao tratada.
    if feature_1.endswith("_zscore") and not feature_2.endswith("_zscore"):
        return feature_1, feature_2, "Manter versão zscore e remover variável original redundante."

    if feature_2.endswith("_zscore") and not feature_1.endswith("_zscore"):
        return feature_2, feature_1, "Manter versão zscore e remover variável original redundante."

    if feature_1.endswith("_minmax") and not feature_2.endswith("_minmax"):
        return feature_1, feature_2, "Manter versão minmax e remover variável original redundante."

    if feature_2.endswith("_minmax") and not feature_1.endswith("_minmax"):
        return feature_2, feature_1, "Manter versão minmax e remover variável original redundante."

    # se uma feature e das principais e a outra nao, mantem a principal
    if feature_1 in regras_manter and feature_2 not in regras_manter:
        return feature_1, feature_2, "Manter feature metodológica principal."

    if feature_2 in regras_manter and feature_1 not in regras_manter:
        return feature_2, feature_1, "Manter feature metodológica principal."

    return "", "", "Revisar manualmente antes de remover."


def adicionar_decisoes(pares_df):
    if pares_df.empty:
        pares_df["manter_sugerido"] = []
        pares_df["remover_sugerido"] = []
        pares_df["decisao"] = []
        return pares_df

    decisoes = []

    # aplica a heuristica de decisao pra cada par correlacionado
    for _, row in pares_df.iterrows():
        manter, remover, decisao = sugerir_decisao(
            row["feature_1"],
            row["feature_2"],
        )

        decisoes.append(
            {
                "manter_sugerido": manter,
                "remover_sugerido": remover,
                "decisao": decisao,
            }
        )

    decisoes_df = pd.DataFrame(decisoes)

    # cola as decisões no dataframe de pares
    pares_df = pd.concat([pares_df, decisoes_df], axis=1)

    return pares_df


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df_original = carregar_dataset()
    validar_features_finais(df_original)

    df_tratado, resumo_nan_antes, resumo_nan_depois = tratar_nan(df_original)

    features = selecionar_features_numericas(df_tratado)

    # calcula as duas matrizes de correlacao
    corr = gerar_matriz_correlacao(df_tratado, features)
    corr_target = gerar_correlacao_target(df_tratado, features)

    pares_df = identificar_correlacoes_altas(corr, limite=0.85)
    pares_df = adicionar_decisoes(pares_df)

    # salva o dataset sem NaN e os artefatos de correlacao
    df_tratado.to_csv(OUTPUT_DATASET, index=False)
    corr.to_csv(OUTPUT_CORRELATION_MATRIX)
    corr_target.to_csv(OUTPUT_TARGET_CORRELATION, index=False)
    salvar_heatmap_correlacao(corr)

    # se nao encontrou pares, cria dataframe vazio com as colunas certas
    if pares_df.empty:
        pares_df = pd.DataFrame(
            columns=[
                "feature_1",
                "feature_2",
                "correlacao",
                "correlacao_abs",
                "manter_sugerido",
                "remover_sugerido",
                "decisao",
            ]
        )

    pares_df.to_csv(OUTPUT_HIGH_CORRELATION, index=False)

    print("Análise de correlação concluída com sucesso.")
    print(f"Dataset sem NaN: {OUTPUT_DATASET}")
    print(f"Matriz de correlação: {OUTPUT_CORRELATION_MATRIX}")
    print(f"Heatmap de correlação: {OUTPUT_CORRELATION_MATRIX_PNG}")
    print(f"Pares com correlação alta: {OUTPUT_HIGH_CORRELATION}")


if __name__ == "__main__":
    main()
