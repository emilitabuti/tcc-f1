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
OUTPUT_FEATURE_LIST = MODELS_DIR / "features_modelagem_2018_2025.json"
OUTPUT_REPORT = MODELS_DIR / "relatorio_13_selecao_features_modelagem.txt"

TARGET = "finish_position"


COLUNAS_REMOVER_FIXAS = [
    # Identificação / texto
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

    # Target ou variáveis diretamente ligadas ao resultado final
    "points",

    # Colunas auxiliares que não devem entrar como feature
    "driver_win_flag",
]


# Remoções manuais iniciais por redundância.
# Ajuste esta lista depois de revisar o arquivo pares_correlacao_alta_maior_085.csv.
COLUNAS_REMOVER_CORRELACAO_MANUAL = [
    # versões originais quando já existem normalizadas
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

    # versões estáticas se existir versão dinâmica
    "track_complexity_static",
    "avg_pit_stops_circuit_static_global",
]


FEATURES_PRIORITARIAS_MANTER = [
    "grid_position_minmax",
    "laps_minmax",
    "compound_ordinal",
    "track_complexity",
    "weather_impact_factor",
    "safety_car_flag",
    "avg_pit_stops_circuit",

    "driver_coef_rapm",
    "constructor_coef_rapm",
    "recent_form_5",
    "recent_form_3",
    "recent_form_cold_start_flag",
    "driver_experience",
    "driver_wins_total",
    "constructor_wins_total",
    "driver_dnf_rate",
    "constructor_dnf_rate",
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


def selecionar_features(df, pares_df):
    colunas_remover = set()

    for col in COLUNAS_REMOVER_FIXAS:
        if col in df.columns:
            colunas_remover.add(col)

    for col in COLUNAS_REMOVER_CORRELACAO_MANUAL:
        if col in df.columns:
            colunas_remover.add(col)

    remocoes_sugeridas = obter_remocoes_sugeridas(pares_df)

    for col in remocoes_sugeridas:
        if col in df.columns and col not in FEATURES_PRIORITARIAS_MANTER:
            colunas_remover.add(col)

    # Nunca remove o target aqui.
    if TARGET in colunas_remover:
        colunas_remover.remove(TARGET)

    colunas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()

    features = [
        col for col in colunas_numericas
        if col != TARGET and col not in colunas_remover
    ]

    # Garante que features prioritárias entrem se existirem.
    for col in FEATURES_PRIORITARIAS_MANTER:
        if col in df.columns and col not in features and col != TARGET:
            features.append(col)

    # Remove duplicidade preservando ordem.
    features = list(dict.fromkeys(features))

    colunas_saida = [TARGET] + features

    df_modelagem = df[colunas_saida].copy()

    return df_modelagem, features, sorted(colunas_remover)


def gerar_relatorio(df_original, df_modelagem, features, colunas_removidas, pares_df):
    linhas = []

    linhas.append("Relatório 13 - Seleção Final de Features para Modelagem")
    linhas.append("=" * 65)
    linhas.append("")
    linhas.append(f"Dataset de entrada: {INPUT_DATASET}")
    linhas.append(f"Dataset final de modelagem: {OUTPUT_DATASET}")
    linhas.append(f"Linhas: {len(df_modelagem)}")
    linhas.append(f"Colunas originais: {df_original.shape[1]}")
    linhas.append(f"Features finais: {len(features)}")
    linhas.append(f"Target: {TARGET}")
    linhas.append("")

    linhas.append("Features finais selecionadas")
    linhas.append("-" * 35)
    for feature in features:
        linhas.append(f"- {feature}")

    linhas.append("")
    linhas.append("Colunas removidas")
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
        "Foram removidas colunas textuais, identificadores, variáveis auxiliares "
        "e variáveis redundantes com versões normalizadas. "
        "As features metodológicas principais foram preservadas para a etapa de modelagem."
    )

    OUTPUT_REPORT.write_text("\n".join(linhas), encoding="utf-8")


def salvar_lista_features(features):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "target": TARGET,
        "n_features": len(features),
        "features": features,
    }

    OUTPUT_FEATURE_LIST.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = carregar_dataset()
    pares_df = carregar_pares_correlacao()

    df_modelagem, features, colunas_removidas = selecionar_features(df, pares_df)

    if df_modelagem.isna().sum().sum() > 0:
        raise RuntimeError("O dataset final de modelagem ainda possui NaN.")

    OUTPUT_DATASET.parent.mkdir(parents=True, exist_ok=True)

    df_modelagem.to_csv(OUTPUT_DATASET, index=False)

    salvar_lista_features(features)

    gerar_relatorio(
        df_original=df,
        df_modelagem=df_modelagem,
        features=features,
        colunas_removidas=colunas_removidas,
        pares_df=pares_df,
    )

    print("Seleção final de features concluída com sucesso.")
    print(f"Dataset de modelagem: {OUTPUT_DATASET}")
    print(f"Lista de features: {OUTPUT_FEATURE_LIST}")
    print(f"Relatório: {OUTPUT_REPORT}")
    print(f"Total de features finais: {len(features)}")


if __name__ == "__main__":
    main()