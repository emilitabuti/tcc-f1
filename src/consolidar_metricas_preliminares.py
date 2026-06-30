from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = BASE_DIR / "reports" / "modelagem"

# caminhos dos csvs gerados pelo walk-forward de cada modelo
INPUTS = {
    "xgboost": REPORTS_DIR / "metricas_walk_forward_xgboost.csv",
    "random_forest": REPORTS_DIR / "metricas_walk_forward_random_forest.csv",
    "lightgbm": REPORTS_DIR / "metricas_walk_forward_lightgbm.csv",
}

OUTPUT_TABELA = REPORTS_DIR / "tabela_metricas_preliminares_3modelos.csv"
OUTPUT_RESUMO = REPORTS_DIR / "tabela_metricas_preliminares_3modelos_resumo.csv"

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
        # checa se todas as colunas que a gente precisa estão lá
        faltantes = sorted(set(COLUNAS_ESPERADAS).difference(df.columns))
        if faltantes:
            raise ValueError(f"Colunas ausentes em {caminho}: {faltantes}")

        df = df[COLUNAS_ESPERADAS].copy()
        df.insert(0, "modelo", modelo)
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def criar_resumo(df: pd.DataFrame) -> pd.DataFrame:
    # agrega as métricas por modelo, calcula média e desvio
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

    # pega qual fold teve o melhor e pior MAE pra cada modelo
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


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = carregar_metricas()
    resumo = criar_resumo(df)

    # salva a tabela completa e o resumo em csv separados
    df.to_csv(OUTPUT_TABELA, index=False, encoding="utf-8-sig")
    resumo.to_csv(OUTPUT_RESUMO, index=False, encoding="utf-8-sig")

    print("Consolidacao preliminar concluida.")
    print(resumo.to_string(index=False))


if __name__ == "__main__":
    main()
