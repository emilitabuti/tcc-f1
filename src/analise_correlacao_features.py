from pathlib import Path
import numpy as np
import pandas as pd


# 12 - Análise de correlação e tratamento final de NaN
#
# Objetivo:
# - integrar e validar o dataset final de features;
# - tratar NaN gerados no feature engineering;
# - gerar matriz de correlação entre features numéricas;
# - identificar pares com correlação alta, acima de 0.85;
# - documentar decisões iniciais de remoção.


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports" / "correlacao_features"

INPUT_FILE = PROCESSED_DIR / "dataset_features_final_2018_2025.csv"

OUTPUT_DATASET = PROCESSED_DIR / "dataset_features_final_2018_2025_sem_nan.csv"
OUTPUT_CORRELATION_MATRIX = REPORTS_DIR / "matriz_correlacao_features.csv"
OUTPUT_HIGH_CORRELATION = REPORTS_DIR / "pares_correlacao_alta_maior_085.csv"
OUTPUT_REPORT = REPORTS_DIR / "relatorio_correlacao_features.txt"


TARGET_COLUMNS = [
    "finish_position",
]


ID_COLUMNS = [
    "season",
    "round",
    "race_name",
    "driver_id",
    "constructor_id",
    "RaceID",
    "status",
    "status_normalizado",
    "dnf_categoria",
    "circuit_id",
    "compound_normalizado",
    "circuit_type",
    "outlier_colunas",
    "outlier_tipo",
]


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
    "avg_pit_stops_circuit",
    "track_complexity_static",
    "incident_rate_hist",
    "incident_rate_hist_norm",
]


def carregar_dataset():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    if "RaceID" in df.columns:
        duplicados = df["RaceID"].duplicated().sum()
        if duplicados > 0:
            raise RuntimeError(f"Foram encontrados {duplicados} RaceID duplicados.")

    return df


def tratar_nan(df):
    df = df.copy()

    resumo_nan_antes = df.isna().sum()
    resumo_nan_antes = resumo_nan_antes[resumo_nan_antes > 0].sort_values(ascending=False)

    for col in FEATURES_NAN_ZERO:
        if col in df.columns:
            df[col] = df[col].fillna(0)

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

    resumo_nan_depois = df.isna().sum()
    resumo_nan_depois = resumo_nan_depois[resumo_nan_depois > 0].sort_values(ascending=False)

    return df, resumo_nan_antes, resumo_nan_depois


def selecionar_features_numericas(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    excluir = set(TARGET_COLUMNS)

    features = [
        col for col in numeric_cols
        if col not in excluir
    ]

    return features


def gerar_matriz_correlacao(df, features):
    corr = df[features].corr(method="pearson")

    return corr


def identificar_correlacoes_altas(corr, limite=0.85):
    pares = []

    colunas = corr.columns.tolist()

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
    regras_manter = [
        "driver_coef_rapm",
        "constructor_coef_rapm",
        "recent_form_5",
        "recent_form_3",
        "driver_experience",
        "driver_wins_total",
        "constructor_wins_total",
        "driver_dnf_rate",
        "constructor_dnf_rate",
        "driver_constructor_synergy",
        "track_complexity",
        "weather_impact_factor",
        "avg_pit_stops_circuit",
        "grid_position_minmax",
        "laps_minmax",
    ]

    # Se uma é versão normalizada/zscore e a outra é original,
    # sugere manter a versão tratada.
    if feature_1.endswith("_zscore") and not feature_2.endswith("_zscore"):
        return feature_1, feature_2, "Manter versão zscore e remover variável original redundante."

    if feature_2.endswith("_zscore") and not feature_1.endswith("_zscore"):
        return feature_2, feature_1, "Manter versão zscore e remover variável original redundante."

    if feature_1.endswith("_minmax") and not feature_2.endswith("_minmax"):
        return feature_1, feature_2, "Manter versão minmax e remover variável original redundante."

    if feature_2.endswith("_minmax") and not feature_1.endswith("_minmax"):
        return feature_2, feature_1, "Manter versão minmax e remover variável original redundante."

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

    pares_df = pd.concat([pares_df, decisoes_df], axis=1)

    return pares_df


def gerar_relatorio(
    df_original,
    df_tratado,
    features,
    pares_df,
    resumo_nan_antes,
    resumo_nan_depois,
):
    linhas = []

    linhas.append("Relatório 12 - Tratamento de NaN e Análise de Correlação")
    linhas.append("=" * 65)
    linhas.append("")
    linhas.append(f"Arquivo de entrada: {INPUT_FILE}")
    linhas.append(f"Arquivo tratado gerado: {OUTPUT_DATASET}")
    linhas.append(f"Linhas do dataset: {len(df_tratado)}")
    linhas.append(f"Colunas do dataset: {df_tratado.shape[1]}")
    linhas.append(f"Features numéricas analisadas: {len(features)}")
    linhas.append("")

    linhas.append("Tratamento de NaN")
    linhas.append("-" * 20)

    if resumo_nan_antes.empty:
        linhas.append("Não havia NaN no dataset de entrada.")
    else:
        linhas.append("Colunas com NaN antes do tratamento:")
        for col, qtd in resumo_nan_antes.items():
            linhas.append(f"- {col}: {qtd}")

    linhas.append("")

    if resumo_nan_depois.empty:
        linhas.append("Após o tratamento, não restaram valores NaN.")
    else:
        linhas.append("Ainda existem NaN após o tratamento:")
        for col, qtd in resumo_nan_depois.items():
            linhas.append(f"- {col}: {qtd}")

    linhas.append("")
    linhas.append("Análise de correlação")
    linhas.append("-" * 22)
    linhas.append("Foi calculada a matriz de correlação de Pearson entre todas as features numéricas.")
    linhas.append("Foram destacados pares com correlação absoluta maior que 0.85.")
    linhas.append(f"Total de pares com correlação alta: {len(pares_df)}")
    linhas.append("")

    if not pares_df.empty:
        linhas.append("Principais pares encontrados:")
        top = pares_df.head(30)

        for _, row in top.iterrows():
            linhas.append(
                f"- {row['feature_1']} x {row['feature_2']}: "
                f"r={row['correlacao']:.4f} | decisão: {row['decisao']}"
            )

    linhas.append("")
    linhas.append("Observação metodológica")
    linhas.append("-" * 24)
    linhas.append(
        "A remoção automática não foi aplicada nesta etapa. "
        "O arquivo de pares altamente correlacionados foi gerado para revisão e documentação "
        "antes da definição final das features de modelagem."
    )

    OUTPUT_REPORT.write_text("\n".join(linhas), encoding="utf-8")


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df_original = carregar_dataset()

    df_tratado, resumo_nan_antes, resumo_nan_depois = tratar_nan(df_original)

    features = selecionar_features_numericas(df_tratado)

    corr = gerar_matriz_correlacao(df_tratado, features)

    pares_df = identificar_correlacoes_altas(corr, limite=0.85)
    pares_df = adicionar_decisoes(pares_df)

    df_tratado.to_csv(OUTPUT_DATASET, index=False)
    corr.to_csv(OUTPUT_CORRELATION_MATRIX)

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

    gerar_relatorio(
        df_original=df_original,
        df_tratado=df_tratado,
        features=features,
        pares_df=pares_df,
        resumo_nan_antes=resumo_nan_antes,
        resumo_nan_depois=resumo_nan_depois,
    )

    print("Análise de correlação concluída com sucesso.")
    print(f"Dataset sem NaN: {OUTPUT_DATASET}")
    print(f"Matriz de correlação: {OUTPUT_CORRELATION_MATRIX}")
    print(f"Pares com correlação alta: {OUTPUT_HIGH_CORRELATION}")
    print(f"Relatório: {OUTPUT_REPORT}")


if __name__ == "__main__":
    main()