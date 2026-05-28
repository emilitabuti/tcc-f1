from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = BASE_DIR / "reports" / "modelagem"

INPUTS = {
    "xgboost_tuned": REPORTS_DIR / "metricas_walk_forward_xgboost_tuned.csv",
    "random_forest_tuned": REPORTS_DIR / "metricas_walk_forward_randomforest_tuned.csv",
    "lightgbm_tuned": REPORTS_DIR / "metricas_walk_forward_lightgbm_tuned.csv",
}

OUTPUT_TABELA = REPORTS_DIR / "tabela_metricas_tunadas_3modelos.csv"
OUTPUT_RESUMO = REPORTS_DIR / "tabela_metricas_tunadas_3modelos_resumo.csv"
OUTPUT_RELATORIO = REPORTS_DIR / "relatorio_modelos_tunados_26_27_05.txt"

METRICAS = ["mae", "rmse", "r2", "kendall_tau", "top3_accuracy"]
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
            rmse_std=("rmse", "std"),
            r2_medio=("r2", "mean"),
            kendall_tau_medio=("kendall_tau", "mean"),
            top3_accuracy_medio=("top3_accuracy", "mean"),
        )
        .sort_values(["mae_medio", "kendall_tau_medio"], ascending=[True, False])
        .reset_index(drop=True)
    )

    return resumo


def gerar_relatorio(df: pd.DataFrame, resumo: pd.DataFrame) -> None:
    melhor = resumo.iloc[0]
    linhas = [
        "Relatorio - Modelos Tunados - 26/05 e 27/05",
        "=" * 54,
        "",
        "Escopo:",
        "- Tuning Optuna do XGBoost conforme 26/05.",
        "- Tuning Optuna do Random Forest conforme 27/05.",
        "- LightGBM acrescentado como terceiro modelo comparavel.",
        "- Hiperparametros escolhidos por MAE medio em 2023-2024.",
        "- Reavaliacao final em 2023, 2024 e 2025 com walk-forward.",
        "",
        "Resumo ordenado por MAE medio:",
        resumo.to_string(index=False),
        "",
        "Metricas por fold:",
        df.to_string(index=False),
        "",
        "Leitura:",
        (
            f"- Melhor MAE medio tunado: {melhor['modelo']} "
            f"({melhor['mae_medio']:.6f} +/- {melhor['mae_std']:.6f})."
        ),
        "- Os resultados tunados devem ser comparados aos preliminares sem tuning antes de escolher finalistas.",
        "",
        "Artefatos gerados:",
        f"- {OUTPUT_TABELA}",
        f"- {OUTPUT_RESUMO}",
        f"- {OUTPUT_RELATORIO}",
    ]

    OUTPUT_RELATORIO.write_text("\n".join(linhas), encoding="utf-8")


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = carregar_metricas()
    resumo = criar_resumo(df)

    df.to_csv(OUTPUT_TABELA, index=False, encoding="utf-8-sig")
    resumo.to_csv(OUTPUT_RESUMO, index=False, encoding="utf-8-sig")
    gerar_relatorio(df, resumo)

    print("Consolidacao de modelos tunados concluida.")
    print(resumo.to_string(index=False))


if __name__ == "__main__":
    main()
