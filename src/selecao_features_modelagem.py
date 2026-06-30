from pathlib import Path
import json
import numpy as np
import pandas as pd

# 13 - Selecao final de features para modelagem

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports" / "correlacao_features"
MODELS_DIR = BASE_DIR / "models" / "feature_selection"

INPUT_DATASET = PROCESSED_DIR / "dataset_features_final_2018_2025_sem_nan.csv"
INPUT_CORRELATION_PAIRS = REPORTS_DIR / "pares_correlacao_alta_maior_085.csv"
INPUT_RFE_MANIFEST = MODELS_DIR / "manifest_rfe_xgboost.json"

OUTPUT_DATASET = PROCESSED_DIR / "dataset_modelagem_2018_2025.csv"
OUTPUT_X_2018_2025 = PROCESSED_DIR / "dataset_modelagem_X_2018_2025.csv"
OUTPUT_Y_2018_2025 = PROCESSED_DIR / "dataset_modelagem_y_2018_2025.csv"
OUTPUT_X_2018_2024 = PROCESSED_DIR / "dataset_modelagem_X_2018_2024.csv"
OUTPUT_Y_2018_2024 = PROCESSED_DIR / "dataset_modelagem_y_2018_2024.csv"
OUTPUT_FEATURE_LIST = MODELS_DIR / "features_modelagem_2018_2025.json"

TARGET = "finish_position"
KEY_COLUMNS = ["RaceID", "season", "round", "driver_id", "constructor_id"]


# features que vão de fato entrar no modelo
FEATURES_FINAIS = [
    "qualifying_position",
    "grid_penalty",
    "recent_form_5",
    "driver_coef_rapm",
    "constructor_coef_rapm",
    "constructor_dnf_rate",
    "constructor_wins_total",
    "driver_constructor_synergy",
    "track_complexity",
    "altitude_m",
    "tire_compound_start",
    "season_factor",
    "incident_rate_hist_norm",
]

# colunas que nao podem entrar em X de jeito nenhum (leak ou pos-corrida)
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

    # só pra garantir que o tratamento de NaN do passo anterior funcionou
    if df.isna().sum().sum() > 0:
        raise RuntimeError("Ainda existem valores NaN no dataset de entrada.")

    return df


def carregar_pares_correlacao():
    if not INPUT_CORRELATION_PAIRS.exists():
        return pd.DataFrame()

    return pd.read_csv(INPUT_CORRELATION_PAIRS)


def carregar_features_selecionadas():
    # tenta pegar a lista do manifest do RFE, senão usa o fallback manual
    if not INPUT_RFE_MANIFEST.exists():
        return FEATURES_FINAIS.copy()

    payload = json.loads(INPUT_RFE_MANIFEST.read_text(encoding="utf-8"))
    features = payload.get("selected_features")

    if not features:
        return FEATURES_FINAIS.copy()

    return list(features)


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


def validar_contrato(df, features):
    # checa se todas as colunas necessárias estão no dataset
    faltantes = [col for col in features + [TARGET] + KEY_COLUMNS if col not in df.columns]

    if faltantes:
        raise ValueError(f"Colunas obrigatórias ausentes no dataset de modelagem: {faltantes}")

    nao_numericas = [
        col for col in features
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col])
    ]

    if nao_numericas:
        raise TypeError(f"Features finais não numéricas: {nao_numericas}")


def selecionar_features(df, pares_df, features_selecionadas):
    validar_contrato(df, features_selecionadas)

    features = features_selecionadas.copy()

    # tudo que nao e feature nem target nem chave vai pra lista de removidas
    colunas_removidas = [
        col for col in df.columns
        if col not in set(features + [TARGET] + KEY_COLUMNS)
    ]

    df_x = df[features].copy()
    df_y = df[KEY_COLUMNS + [TARGET]].copy()
    # junta y e X num dataset único pra facilitar o uso depois
    df_modelagem = pd.concat([df_y, df_x], axis=1)

    return df_modelagem, df_x, df_y, features, sorted(colunas_removidas)


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
    features_selecionadas = carregar_features_selecionadas()

    df_modelagem, df_x, df_y, features, colunas_removidas = selecionar_features(
        df,
        pares_df,
        features_selecionadas,
    )

    # validações finais pra garantir que nada furou o contrato
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

    # salva o dataset completo e os splits X/y
    df_modelagem.to_csv(OUTPUT_DATASET, index=False)
    df_x.to_csv(OUTPUT_X_2018_2025, index=False)
    df_y.to_csv(OUTPUT_Y_2018_2025, index=False)

    # versão só até 2024 pra treino sem ver 2025
    mask_2024 = df["season"] <= 2024
    df.loc[mask_2024, features].to_csv(OUTPUT_X_2018_2024, index=False)
    df.loc[mask_2024, KEY_COLUMNS + [TARGET]].to_csv(OUTPUT_Y_2018_2024, index=False)

    salvar_lista_features(features)

    print("Seleção final de features concluída com sucesso.")
    print(f"Dataset combinado de modelagem: {OUTPUT_DATASET}")
    print(f"Dataset X 2018-2025: {OUTPUT_X_2018_2025}")
    print(f"Dataset y 2018-2025: {OUTPUT_Y_2018_2025}")
    print(f"Lista de features: {OUTPUT_FEATURE_LIST}")
    print(f"Total de features finais: {len(features)}")


if __name__ == "__main__":
    main()
