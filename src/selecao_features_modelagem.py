from pathlib import Path
import json
import numpy as np
import pandas as pd


# 13 - Seleção final de features para modelagem
#
# Objetivo:
# - usar o dataset tratado sem NaN;
# - aplicar decisões de remoção baseadas na análise de correlação;
# - remover colunas não utilizadas como features;
# - separar target e features;
# - gerar dataset final pronto para modelagem.


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports" / "correlacao_features"
MODELS_DIR = BASE_DIR / "models" / "feature_selection"

INPUT_DATASET = PROCESSED_DIR / "dataset_features_final_2018_2025_sem_nan.csv"
INPUT_CORRELATION_PAIRS = REPORTS_DIR / "pares_correlacao_alta_maior_085.csv"

OUTPUT_DATASET = PROCESSED_DIR / "dataset_modelagem_2018_2025.csv"
OUTPUT_X_2018_2025 = PROCESSED_DIR / "dataset_modelagem_X_2018_2025.csv"
OUTPUT_Y_2018_2025 = PROCESSED_DIR / "dataset_modelagem_y_2018_2025.csv"
OUTPUT_X_2018_2024 = PROCESSED_DIR / "dataset_modelagem_X_2018_2024.csv"
OUTPUT_Y_2018_2024 = PROCESSED_DIR / "dataset_modelagem_y_2018_2024.csv"
OUTPUT_FEATURE_LIST = MODELS_DIR / "features_modelagem_2018_2025.json"
OUTPUT_REPORT = MODELS_DIR / "relatorio_13_selecao_features_modelagem.txt"
OUTPUT_FINAL_REPORT = PROCESSED_DIR / "relatorio_feature_engineering_final.txt"

TARGET = "finish_position"
KEY_COLUMNS = ["RaceID", "season", "round", "driver_id", "constructor_id"]


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

COLUNAS_PROIBIDAS_X = [
    TARGET,
    "points",
    "race_points",
    "fastest_lap_race",
    "previous_position",
    "status",
    "laps",
]

FEATURES_MANTIDAS_APESAR_CORRELACAO = [
    "recent_form_5",
    "qualifying_position",
    "driver_constructor_synergy",
]


def carregar_dataset():
    if not INPUT_DATASET.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {INPUT_DATASET}")

    df = pd.read_csv(INPUT_DATASET)

    if TARGET not in df.columns:
        raise ValueError(f"Target ausente no dataset: {TARGET}")

    if df.isna().sum().sum() > 0:
        raise RuntimeError("Ainda existem valores NaN no dataset de entrada.")

    return df


def carregar_pares_correlacao():
    if not INPUT_CORRELATION_PAIRS.exists():
        return pd.DataFrame()

    return pd.read_csv(INPUT_CORRELATION_PAIRS)


def obter_remocoes_sugeridas(pares_df):
    if pares_df.empty:
        return []

    if "remover_sugerido" not in pares_df.columns:
        return []

    remover = (
        pares_df["remover_sugerido"]
        .dropna()
        .astype(str)
        .str.strip()
    )

    remover = remover[remover != ""].unique().tolist()

    return remover


def validar_contrato(df):
    faltantes = [col for col in FEATURES_FINAIS + [TARGET] + KEY_COLUMNS if col not in df.columns]

    if faltantes:
        raise ValueError(f"Colunas obrigatórias ausentes no dataset de modelagem: {faltantes}")

    nao_numericas = [
        col for col in FEATURES_FINAIS
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col])
    ]

    if nao_numericas:
        raise TypeError(f"Features finais não numéricas: {nao_numericas}")


def selecionar_features(df, pares_df):
    validar_contrato(df)

    features = FEATURES_FINAIS.copy()
    colunas_removidas = [
        col for col in df.columns
        if col not in set(features + [TARGET] + KEY_COLUMNS)
    ]

    df_x = df[features].copy()
    df_y = df[KEY_COLUMNS + [TARGET]].copy()
    df_modelagem = pd.concat([df_y, df_x], axis=1)

    return df_modelagem, df_x, df_y, features, sorted(colunas_removidas)


def gerar_relatorio(df_original, df_modelagem, df_x, df_y, features, colunas_removidas, pares_df):
    linhas = []

    linhas.append("Relatório 13 - Seleção Final de Features para Modelagem")
    linhas.append("=" * 65)
    linhas.append("")
    linhas.append(f"Dataset de entrada: {INPUT_DATASET}")
    linhas.append(f"Dataset combinado de modelagem: {OUTPUT_DATASET}")
    linhas.append(f"Dataset X 2018-2025: {OUTPUT_X_2018_2025}")
    linhas.append(f"Dataset y 2018-2025: {OUTPUT_Y_2018_2025}")
    linhas.append(f"Dataset X 2018-2024: {OUTPUT_X_2018_2024}")
    linhas.append(f"Dataset y 2018-2024: {OUTPUT_Y_2018_2024}")
    linhas.append(f"Linhas: {len(df_modelagem)}")
    linhas.append(f"Colunas originais: {df_original.shape[1]}")
    linhas.append(f"Colunas X: {df_x.shape[1]}")
    linhas.append(f"Colunas y: {df_y.shape[1]}")
    linhas.append(f"Features finais: {len(features)}")
    linhas.append(f"Target: {TARGET}")
    linhas.append(f"Target presente em X: {TARGET in df_x.columns}")
    linhas.append("")

    linhas.append("Features finais selecionadas")
    linhas.append("-" * 35)
    for feature in features:
        linhas.append(f"- {feature}")

    linhas.append("")
    linhas.append("Colunas mantidas fora de X")
    linhas.append("-" * 20)
    if colunas_removidas:
        for col in colunas_removidas:
            linhas.append(f"- {col}")
    else:
        linhas.append("Nenhuma coluna removida por regra.")

    linhas.append("")
    linhas.append("Resumo da correlação")
    linhas.append("-" * 25)

    if pares_df.empty:
        linhas.append("Arquivo de pares correlacionados não encontrado ou vazio.")
    else:
        linhas.append(f"Pares com correlação alta no relatório anterior: {len(pares_df)}")

        if "remover_sugerido" in pares_df.columns:
            remocoes = obter_remocoes_sugeridas(pares_df)
            linhas.append(f"Colunas com remoção sugerida: {len(remocoes)}")

    linhas.append("")
    linhas.append("Decisão metodológica")
    linhas.append("-" * 25)
    linhas.append(
        "O dataset X foi corrigido para conter apenas features disponíveis antes da corrida. "
        "safety_car_flag foi mantida como auditoria fora de X e substituída por "
        "incident_rate_hist_norm. weather_impact_factor foi recalculada como histórico causal "
        "por circuito, mas ficou fora do X final após RFE. Também foram removidas "
        "recent_form_3 e grid_position por redundância empírica com recent_form_5 e "
        "qualifying_position. A RFE temporal com XGBoost selecionou 15 features finais."
    )

    linhas.append("")
    linhas.append("Features mantidas apesar de correlação alta")
    linhas.append("-" * 44)
    for feature in FEATURES_MANTIDAS_APESAR_CORRELACAO:
        linhas.append(f"- {feature}")

    conteudo = "\n".join(linhas)
    OUTPUT_REPORT.write_text(conteudo, encoding="utf-8")
    OUTPUT_FINAL_REPORT.write_text(conteudo, encoding="utf-8")


def salvar_lista_features(features):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "target": TARGET,
        "n_features": len(features),
        "features": features,
        "key_columns_y": KEY_COLUMNS,
        "forbidden_columns_x": COLUNAS_PROIBIDAS_X,
    }

    OUTPUT_FEATURE_LIST.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = carregar_dataset()
    pares_df = carregar_pares_correlacao()

    df_modelagem, df_x, df_y, features, colunas_removidas = selecionar_features(df, pares_df)

    if df_modelagem.isna().sum().sum() > 0:
        raise RuntimeError("O dataset final de modelagem ainda possui NaN.")
    if df_x.isna().sum().sum() > 0 or df_y.isna().sum().sum() > 0:
        raise RuntimeError("Os datasets X/y ainda possuem NaN.")
    if TARGET in df_x.columns:
        raise RuntimeError("O target entrou indevidamente em X.")
    proibidas_em_x = [col for col in COLUNAS_PROIBIDAS_X if col in df_x.columns]
    if proibidas_em_x:
        raise RuntimeError(f"Colunas proibidas encontradas em X: {proibidas_em_x}")

    OUTPUT_DATASET.parent.mkdir(parents=True, exist_ok=True)

    df_modelagem.to_csv(OUTPUT_DATASET, index=False)
    df_x.to_csv(OUTPUT_X_2018_2025, index=False)
    df_y.to_csv(OUTPUT_Y_2018_2025, index=False)

    mask_2024 = df["season"] <= 2024
    df.loc[mask_2024, features].to_csv(OUTPUT_X_2018_2024, index=False)
    df.loc[mask_2024, KEY_COLUMNS + [TARGET]].to_csv(OUTPUT_Y_2018_2024, index=False)

    salvar_lista_features(features)

    gerar_relatorio(
        df_original=df,
        df_modelagem=df_modelagem,
        df_x=df_x,
        df_y=df_y,
        features=features,
        colunas_removidas=colunas_removidas,
        pares_df=pares_df,
    )

    print("Seleção final de features concluída com sucesso.")
    print(f"Dataset combinado de modelagem: {OUTPUT_DATASET}")
    print(f"Dataset X 2018-2025: {OUTPUT_X_2018_2025}")
    print(f"Dataset y 2018-2025: {OUTPUT_Y_2018_2025}")
    print(f"Lista de features: {OUTPUT_FEATURE_LIST}")
    print(f"Relatório: {OUTPUT_REPORT}")
    print(f"Total de features finais: {len(features)}")


if __name__ == "__main__":
    main()
