from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = BASE_DIR / "reports" / "modelagem"

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
OUTPUT_RELATORIO = REPORTS_DIR / "relatorio_modelos_tunados_26_28_05.txt"
OUTPUT_DECISAO = REPORTS_DIR / "decisao_preliminar_algoritmos.md"

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

    resumo["melhor_fold"] = resumo["modelo"].map(
        df.loc[df.groupby("modelo")["mae"].idxmin()].set_index("modelo")["valid_season"]
    )
    resumo["pior_fold"] = resumo["modelo"].map(
        df.loc[df.groupby("modelo")["mae"].idxmax()].set_index("modelo")["valid_season"]
    )
    resumo["tempo_tuning_segundos"] = resumo["modelo"].map(carregar_tempos_tuning())
    resumo["tempo_tuning_minutos"] = resumo["tempo_tuning_segundos"].apply(
        lambda valor: pd.NA if pd.isna(valor) else float(valor) / 60
    )

    return resumo


def carregar_tempos_tuning() -> dict:
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


def descrever_tempos_tuning(resumo: pd.DataFrame) -> str:
    ausentes = resumo.loc[
        resumo["tempo_tuning_segundos"].isna(),
        "modelo",
    ].tolist()

    if not ausentes:
        return "- Tempo de tuning registrado para todos os modelos."

    modelos = ", ".join(ausentes)
    return (
        "- Tempo de tuning ausente para execucoes antigas sem instrumentacao: "
        f"{modelos}."
    )


def gerar_relatorio(df: pd.DataFrame, resumo: pd.DataFrame) -> None:
    melhor = resumo.iloc[0]
    ensembles_com_tempo = resumo[
        (resumo["modelo"] != "ridge_baseline")
        & resumo["tempo_tuning_segundos"].notna()
    ]
    mais_rapido = (
        ensembles_com_tempo.sort_values("tempo_tuning_segundos").iloc[0]
        if not ensembles_com_tempo.empty
        else None
    )

    linhas = [
        "Relatorio - Modelos Tunados - 26/05 a 28/05",
        "=" * 54,
        "",
        "Escopo:",
        "- Tuning Optuna do XGBoost conforme 26/05.",
        "- Tuning Optuna do Random Forest conforme 27/05.",
        "- LightGBM acrescentado como terceiro modelo comparavel.",
        "- Ridge Regression incluido como baseline linear com StandardScaler e time-decay.",
        "- Hiperparametros escolhidos por MAE medio em 2023-2024.",
        "- Reavaliacao final em 2023, 2024 e 2025 com walk-forward.",
        descrever_tempos_tuning(resumo),
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
    ]

    if mais_rapido is not None:
        linhas.append(
            f"- Modelo de arvore com menor tempo de tuning: {mais_rapido['modelo']} "
            f"({mais_rapido['tempo_tuning_segundos']:.2f} segundos / "
            f"{mais_rapido['tempo_tuning_minutos']:.2f} minutos)."
        )

    linhas.extend([
        "- Esta consolidacao fecha a quinta-feira com metricas finais preliminares dos 4 modelos.",
        "",
        "Artefatos gerados:",
        f"- {OUTPUT_TABELA_4}",
        f"- {OUTPUT_RESUMO_4}",
        f"- {OUTPUT_DECISAO}",
        f"- {OUTPUT_RELATORIO}",
    ])

    OUTPUT_RELATORIO.write_text("\n".join(linhas), encoding="utf-8")


def gerar_decisao_preliminar(resumo: pd.DataFrame) -> None:
    modelos_sem_ridge = resumo[resumo["modelo"] != "ridge_baseline"].copy()
    finalistas = modelos_sem_ridge.head(2)
    descartado = modelos_sem_ridge.tail(1).iloc[0]
    ridge = resumo[resumo["modelo"] == "ridge_baseline"].iloc[0]

    linhas = [
        "# Decisao Preliminar dos Algoritmos Finalistas",
        "",
        "## Resultado",
        "",
        "Finalistas preliminares para a Fase 1:",
        "",
    ]

    for _, row in finalistas.iterrows():
        linhas.append(
            f"- {row['modelo']}: MAE medio {row['mae_medio']:.4f}, "
            f"Kendall tau {row['kendall_tau_medio']:.4f}, "
            f"MAE std {row['mae_std']:.4f}."
        )

    linhas.extend(
        [
            "",
            "Modelo arquivado como terceiro candidato:",
            "",
            (
                f"- {descartado['modelo']}: MAE medio {descartado['mae_medio']:.4f}, "
                f"Kendall tau {descartado['kendall_tau_medio']:.4f}, "
                f"MAE std {descartado['mae_std']:.4f}."
            ),
            "",
            "Baseline linear:",
            "",
            (
                f"- {ridge['modelo']}: MAE medio {ridge['mae_medio']:.4f}, "
                f"Kendall tau {ridge['kendall_tau_medio']:.4f}. "
                "Permanece como referencia metodologica, nao como finalista principal."
            ),
            "",
            "## Justificativa",
            "",
            (
                "A escolha segue o criterio definido no cronograma revisado: menor MAE medio, "
                "maior Kendall tau, estabilidade entre folds e coerencia com a arquitetura. "
                "O baseline Ridge foi mantido como referencia linear forte baseada na "
                "fundamentacao RAPM; como ele ficou competitivo, os modelos de arvore devem "
                "ser justificados tambem pela analise de importancia de features, robustez e "
                "uso posterior na Fase 2 de drift/adaptacao."
            ),
            "",
            "A decisao ainda deve ser confirmada apos a etapa de feature selection e feature importance.",
        ]
    )

    OUTPUT_DECISAO.write_text("\n".join(linhas), encoding="utf-8")


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = carregar_metricas()
    resumo = criar_resumo(df)
    df_sem_ridge = df[df["modelo"] != "ridge_baseline"].copy()
    resumo_sem_ridge = criar_resumo(df_sem_ridge)

    df_sem_ridge.to_csv(OUTPUT_TABELA, index=False, encoding="utf-8-sig")
    resumo_sem_ridge.to_csv(OUTPUT_RESUMO, index=False, encoding="utf-8-sig")
    df.to_csv(OUTPUT_TABELA_4, index=False, encoding="utf-8-sig")
    resumo.to_csv(OUTPUT_RESUMO_4, index=False, encoding="utf-8-sig")
    gerar_relatorio(df, resumo)
    gerar_decisao_preliminar(resumo)

    print("Consolidacao de modelos tunados concluida.")
    print(resumo.to_string(index=False))


if __name__ == "__main__":
    main()
