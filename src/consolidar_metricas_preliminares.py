from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = BASE_DIR / "reports" / "modelagem"

INPUTS = {
    "xgboost": REPORTS_DIR / "metricas_walk_forward_xgboost.csv",
    "random_forest": REPORTS_DIR / "metricas_walk_forward_random_forest.csv",
    "lightgbm": REPORTS_DIR / "metricas_walk_forward_lightgbm.csv",
}

OUTPUT_TABELA = REPORTS_DIR / "tabela_metricas_preliminares_3modelos.csv"
OUTPUT_RESUMO = REPORTS_DIR / "tabela_metricas_preliminares_3modelos_resumo.csv"
OUTPUT_RELATORIO = REPORTS_DIR / "relatorio_terca_semana2_modelos_preliminares.txt"

METRICAS = ["mae", "rmse", "r2", "kendall_tau"]
COLUNAS_ESPERADAS = METRICAS + [
    "train_until",
    "valid_season",
    "decay",
    "n_train",
    "n_valid",
]


def carregar_metricas() -> pd.DataFrame:
    frames = []

    for modelo, caminho in INPUTS.items():
        if not caminho.exists():
            raise FileNotFoundError(f"Metricas ausentes para {modelo}: {caminho}")

        df = pd.read_csv(caminho)
        faltantes = sorted(set(COLUNAS_ESPERADAS).difference(df.columns))
        if faltantes:
            raise ValueError(f"Colunas ausentes em {caminho}: {faltantes}")

        df = df[COLUNAS_ESPERADAS].copy()
        df.insert(0, "modelo", modelo)
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def criar_resumo(df: pd.DataFrame) -> pd.DataFrame:
    resumo = (
        df.groupby("modelo", as_index=False)
        .agg(
            mae_medio=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_medio=("rmse", "mean"),
            r2_medio=("r2", "mean"),
            kendall_tau_medio=("kendall_tau", "mean"),
        )
    )

    melhor_fold = (
        df.loc[df.groupby("modelo")["mae"].idxmin(), ["modelo", "valid_season"]]
        .rename(columns={"valid_season": "melhor_fold"})
    )
    pior_fold = (
        df.loc[df.groupby("modelo")["mae"].idxmax(), ["modelo", "valid_season"]]
        .rename(columns={"valid_season": "pior_fold"})
    )

    resumo = (
        resumo.merge(melhor_fold, on="modelo")
        .merge(pior_fold, on="modelo")
        .sort_values(["mae_medio", "kendall_tau_medio"], ascending=[True, False])
        .reset_index(drop=True)
    )

    return resumo


def gerar_relatorio(df: pd.DataFrame, resumo: pd.DataFrame):
    melhor = resumo.iloc[0]
    linhas = [
        "Relatorio - Terca Semana 2 - Modelos Preliminares",
        "=" * 58,
        "",
        "Escopo:",
        "- Comparacao sem tuning Optuna.",
        "- Modelos: XGBoost, Random Forest e LightGBM.",
        "- Folds: 2018-2022 -> 2023, 2018-2023 -> 2024, 2018-2024 -> 2025.",
        "- Time-decay aplicado via sample_weight.",
        "",
        "Arquivos consolidados:",
    ]

    for modelo, caminho in INPUTS.items():
        linhas.append(f"- {modelo}: {caminho}")

    linhas.extend(
        [
            "",
            "Resumo ordenado por MAE medio:",
            resumo.to_string(index=False),
            "",
            "Metricas por fold:",
            df.to_string(index=False),
            "",
            "Leitura preliminar:",
            (
                f"- Melhor MAE medio sem tuning: {melhor['modelo']} "
                f"({melhor['mae_medio']:.6f})."
            ),
            "- Esta comparacao ainda nao define finalistas; a decisao final depende de Optuna, Ridge baseline, RFE e feature importance.",
            "",
            "Artefatos gerados:",
            f"- {OUTPUT_TABELA}",
            f"- {OUTPUT_RESUMO}",
            f"- {OUTPUT_RELATORIO}",
        ]
    )

    OUTPUT_RELATORIO.write_text("\n".join(linhas), encoding="utf-8")


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = carregar_metricas()
    resumo = criar_resumo(df)

    df.to_csv(OUTPUT_TABELA, index=False, encoding="utf-8-sig")
    resumo.to_csv(OUTPUT_RESUMO, index=False, encoding="utf-8-sig")
    gerar_relatorio(df, resumo)

    print("Consolidacao preliminar concluida.")
    print(resumo.to_string(index=False))


if __name__ == "__main__":
    main()
