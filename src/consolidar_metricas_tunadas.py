from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from tuning_utils import score_composto_metricas


BASE_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = BASE_DIR / "reports" / "modelagem"

# agora com os modelos tunados via Optuna + o Ridge como baseline
INPUTS = {
    "xgboost_tuned": REPORTS_DIR / "metricas_walk_forward_xgboost_tuned.csv",
    "random_forest_tuned": REPORTS_DIR / "metricas_walk_forward_randomforest_tuned.csv",
    "lightgbm_tuned": REPORTS_DIR / "metricas_walk_forward_lightgbm_tuned.csv",
    "ridge_baseline": REPORTS_DIR / "metricas_ridge_baseline.csv",
}

OUTPUT_TABELA = REPORTS_DIR / "tabela_metricas_tunadas_3modelos.csv"
OUTPUT_RESUMO = REPORTS_DIR / "tabela_metricas_tunadas_3modelos_resumo.csv"
OUTPUT_TABELA_4 = REPORTS_DIR / "tabela_metricas_tunadas_4modelos.csv"
OUTPUT_RESUMO_4 = REPORTS_DIR / "tabela_metricas_tunadas_4modelos_resumo.csv"

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
    # agrega por modelo e já calcula desvio padrão também
    resumo = (
        df.groupby("modelo", as_index=False)
        .agg(
            mae_medio=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_medio=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            r2_medio=("r2", "mean"),
            kendall_tau_medio=("kendall_tau", "mean"),
        )
        .sort_values(["mae_medio", "kendall_tau_medio"], ascending=[True, False])
        .reset_index(drop=True)
    )

    # score composto multi-métrica - combina MAE, RMSE, R2 e tau num único número
    resumo["score_composto_medio"] = resumo["modelo"].map(
        {
            modelo: score_composto_metricas(grupo)
            for modelo, grupo in df.groupby("modelo")
        }
    )
    resumo = resumo.sort_values(
        ["score_composto_medio", "mae_medio"],
        ascending=[False, True],
    ).reset_index(drop=True)

    # melhor e pior fold por MAE
    resumo["melhor_fold"] = resumo["modelo"].map(
        df.loc[df.groupby("modelo")["mae"].idxmin()].set_index("modelo")["valid_season"]
    )
    resumo["pior_fold"] = resumo["modelo"].map(
        df.loc[df.groupby("modelo")["mae"].idxmax()].set_index("modelo")["valid_season"]
    )

    # tempo de tuning em segundos e minutos, pra ter referencia
    resumo["tempo_tuning_segundos"] = resumo["modelo"].map(carregar_tempos_tuning())
    resumo["tempo_tuning_minutos"] = resumo["tempo_tuning_segundos"].apply(
        lambda valor: pd.NA if pd.isna(valor) else float(valor) / 60
    )

    return resumo


def carregar_tempos_tuning() -> dict:
    # le o tempo de tuning de cada json do Optuna
    tempos = {}
    arquivos = {
        "xgboost_tuned": REPORTS_DIR / "optuna_xgboost_best_params.json",
        "random_forest_tuned": REPORTS_DIR / "optuna_randomforest_best_params.json",
        "lightgbm_tuned": REPORTS_DIR / "optuna_lightgbm_best_params.json",
        "ridge_baseline": REPORTS_DIR / "ridge_best_params.json",
    }

    for modelo, caminho in arquivos.items():
        if not caminho.exists():
            tempos[modelo] = pd.NA
            continue

        dados = json.loads(caminho.read_text(encoding="utf-8"))
        tempos[modelo] = dados.get("tempo_tuning_segundos", pd.NA)

    return tempos


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = carregar_metricas()
    resumo = criar_resumo(df)

    # versão sem o Ridge (só os 3 modelos de árvore) e versão completa com 4
    df_sem_ridge = df[df["modelo"] != "ridge_baseline"].copy()
    resumo_sem_ridge = criar_resumo(df_sem_ridge)

    df_sem_ridge.to_csv(OUTPUT_TABELA, index=False, encoding="utf-8-sig")
    resumo_sem_ridge.to_csv(OUTPUT_RESUMO, index=False, encoding="utf-8-sig")
    df.to_csv(OUTPUT_TABELA_4, index=False, encoding="utf-8-sig")
    resumo.to_csv(OUTPUT_RESUMO_4, index=False, encoding="utf-8-sig")

    print("Consolidacao de modelos tunados concluida.")
    print(resumo.to_string(index=False))


if __name__ == "__main__":
    main()
