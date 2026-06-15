from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = BASE_DIR / "reports" / "modelagem"
FIGURES_DIR = REPORTS_DIR / "figures" / "semana3"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

METRICAS_RESUMO = REPORTS_DIR / "tabela_metricas_tunadas_4modelos_resumo.csv"
METRICAS_FOLDS = REPORTS_DIR / "tabela_metricas_tunadas_4modelos.csv"
FEATURE_IMPORTANCE = {
    "LightGBM": REPORTS_DIR / "feature_importance_lgb.csv",
    "Random Forest": REPORTS_DIR / "feature_importance_rf.csv",
    "XGBoost": REPORTS_DIR / "feature_importance_xgb.csv",
}
OPENF1_2026 = PROCESSED_DIR / "openf1_2026_available.csv"
RELATORIO_2026 = PROCESSED_DIR / "relatorio_update_2026.txt"

OUTPUT_FEATURES_DOMINANTES = REPORTS_DIR / "tabela_features_dominantes_semana3.csv"
OUTPUT_RELATORIO = REPORTS_DIR / "relatorio_semana3_resultados.md"
OUTPUT_2026 = REPORTS_DIR / "validacao_2026_semana3.md"

MODEL_LABELS = {
    "ridge_baseline": "Ridge",
    "lightgbm_tuned": "LightGBM",
    "random_forest_tuned": "Random Forest",
    "xgboost_tuned": "XGBoost",
}
MODEL_ORDER = ["Ridge", "LightGBM", "Random Forest", "XGBoost"]
PALETTE = {
    "Ridge": "#2f6f73",
    "LightGBM": "#4f7db8",
    "Random Forest": "#7a8f3a",
    "XGBoost": "#b45f3c",
}


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def _savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def carregar_metricas() -> tuple[pd.DataFrame, pd.DataFrame]:
    resumo = _read_csv(METRICAS_RESUMO)
    folds = _read_csv(METRICAS_FOLDS)
    resumo["modelo_label"] = resumo["modelo"].map(MODEL_LABELS)
    folds["modelo_label"] = folds["modelo"].map(MODEL_LABELS)
    return resumo, folds


def grafico_barras_metricas(resumo: pd.DataFrame) -> None:
    metricas = [
        ("mae_medio", "MAE medio", "01_mae_medio.png"),
        ("rmse_medio", "RMSE medio", "02_rmse_medio.png"),
        ("r2_medio", "R2 medio", "03_r2_medio.png"),
        ("kendall_tau_medio", "Kendall tau medio", "04_kendall_tau_medio.png"),
        ("score_composto_medio", "Score composto oficial", "05_score_composto.png"),
    ]

    for coluna, titulo, arquivo in metricas:
        plt.figure(figsize=(8, 4.8))
        ax = sns.barplot(
            data=resumo,
            x="modelo_label",
            y=coluna,
            order=MODEL_ORDER,
            palette=PALETTE,
            hue="modelo_label",
            dodge=False,
            legend=False,
        )
        if coluna in {"mae_medio", "rmse_medio"}:
            std_col = coluna.replace("_medio", "_std")
            if std_col in resumo.columns:
                pos = {label: i for i, label in enumerate(MODEL_ORDER)}
                ordered = resumo.set_index("modelo_label").loc[MODEL_ORDER]
                ax.errorbar(
                    [pos[m] for m in MODEL_ORDER],
                    ordered[coluna],
                    yerr=ordered[std_col],
                    fmt="none",
                    ecolor="#263238",
                    elinewidth=1.2,
                    capsize=4,
                )
        ax.set_title(titulo)
        ax.set_xlabel("")
        ax.set_ylabel(titulo)
        ax.bar_label(ax.containers[0], fmt="%.3f", padding=3, fontsize=9)
        _savefig(FIGURES_DIR / arquivo)


def grafico_folds(folds: pd.DataFrame) -> None:
    metricas = [
        ("mae", "MAE por fold temporal", "06_mae_por_fold.png"),
        ("rmse", "RMSE por fold temporal", "07_rmse_por_fold.png"),
        ("r2", "R2 por fold temporal", "08_r2_por_fold.png"),
        ("kendall_tau", "Kendall tau por fold temporal", "09_kendall_por_fold.png"),
    ]

    for coluna, titulo, arquivo in metricas:
        plt.figure(figsize=(8.5, 4.8))
        ax = sns.lineplot(
            data=folds,
            x="valid_season",
            y=coluna,
            hue="modelo_label",
            hue_order=MODEL_ORDER,
            marker="o",
            palette=PALETTE,
            linewidth=2,
        )
        ax.set_title(titulo)
        ax.set_xlabel("Temporada de validacao")
        ax.set_ylabel(titulo.split(" por ")[0])
        ax.set_xticks(sorted(folds["valid_season"].unique()))
        ax.legend(title="Modelo", frameon=False)
        _savefig(FIGURES_DIR / arquivo)


def carregar_importancias() -> pd.DataFrame:
    frames = []
    for modelo, path in FEATURE_IMPORTANCE.items():
        df = _read_csv(path)
        df["modelo_label"] = modelo
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def grafico_importancias(importancias: pd.DataFrame) -> pd.DataFrame:
    dominantes = []
    for modelo in ["LightGBM", "Random Forest", "XGBoost"]:
        df_modelo = (
            importancias[importancias["modelo_label"] == modelo]
            .sort_values("importance_norm_media", ascending=False)
            .head(8)
            .copy()
        )
        dominantes.append(df_modelo)

        plt.figure(figsize=(8, 5.2))
        ax = sns.barplot(
            data=df_modelo.sort_values("importance_norm_media", ascending=True),
            x="importance_norm_media",
            y="feature",
            color=PALETTE[modelo],
        )
        ax.set_title(f"Top features - {modelo}")
        ax.set_xlabel("Importancia normalizada media")
        ax.set_ylabel("")
        _savefig(FIGURES_DIR / f"10_feature_importance_{modelo.lower().replace(' ', '_')}.png")

    df_dominantes = pd.concat(dominantes, ignore_index=True)
    df_dominantes[
        ["modelo_label", "rank", "feature", "importance_type", "importance_norm_media"]
    ].to_csv(OUTPUT_FEATURES_DOMINANTES, index=False, encoding="utf-8-sig")
    return df_dominantes


def escrever_relatorio(resumo: pd.DataFrame, dominantes: pd.DataFrame) -> None:
    melhor_global = resumo.sort_values("score_composto_medio", ascending=False).iloc[0]
    arvores = resumo[resumo["modelo_label"].isin(["LightGBM", "Random Forest", "XGBoost"])]
    melhor_arvore = arvores.sort_values("score_composto_medio", ascending=False).iloc[0]

    top_features = (
        dominantes.sort_values(["modelo_label", "rank"])
        .groupby("modelo_label")
        .head(5)
        .groupby("modelo_label")["feature"]
        .apply(lambda s: ", ".join(s.tolist()))
    )

    linhas = [
        "# Relatorio Semana 3 - Resultados e Visualizacoes",
        "",
        "## Criterio oficial",
        "",
        "A avaliacao oficial usa `finish_position` como target e as metricas MAE, RMSE, R2 e Kendall tau. Top-3 nao faz parte do criterio oficial.",
        "",
        "## Resultado consolidado",
        "",
        f"- Melhor modelo global por score composto: {melhor_global['modelo_label']} (score={melhor_global['score_composto_medio']:.4f}, MAE={melhor_global['mae_medio']:.4f}).",
        f"- Melhor modelo de arvore: {melhor_arvore['modelo_label']} (score={melhor_arvore['score_composto_medio']:.4f}, MAE={melhor_arvore['mae_medio']:.4f}).",
        "- Random Forest e LightGBM ficam muito proximos; XGBoost permanece como comparativo relevante da literatura.",
        "",
        "## Figuras geradas",
        "",
    ]

    for path in sorted(FIGURES_DIR.glob("*.png")):
        linhas.append(f"- `{path.relative_to(BASE_DIR)}`")

    linhas.extend(["", "## Features dominantes", ""])
    for modelo, features in top_features.items():
        linhas.append(f"- {modelo}: {features}.")

    linhas.extend(
        [
            "",
            "## Artefatos tabulares",
            "",
            f"- `{METRICAS_RESUMO.relative_to(BASE_DIR)}`",
            f"- `{METRICAS_FOLDS.relative_to(BASE_DIR)}`",
            f"- `{OUTPUT_FEATURES_DOMINANTES.relative_to(BASE_DIR)}`",
        ]
    )

    OUTPUT_RELATORIO.write_text("\n".join(linhas) + "\n", encoding="utf-8")


def escrever_validacao_2026() -> None:
    linhas = [
        "# Validacao Semana 3 - Dados 2026",
        "",
        "## Status",
        "",
    ]

    if not OPENF1_2026.exists():
        linhas.extend(
            [
                "`data/processed/openf1_2026_available.csv` nao existe. A analise 2026 deve ser tratada como pendente.",
                "",
            ]
        )
    else:
        df = pd.read_csv(OPENF1_2026)
        n_corridas = int(df["round"].nunique()) if "round" in df.columns else 0
        n_linhas = len(df)
        n_pilotos = int(df["driver_id"].nunique()) if "driver_id" in df.columns else 0
        linhas.extend(
            [
                f"`data/processed/openf1_2026_available.csv` existe com {n_linhas} linhas, {n_pilotos} pilotos unicos e {n_corridas} corridas.",
                "",
                "Leitura metodologica: usar 2026 como analise exploratoria de mudanca temporal, nao como resultado principal da Fase 1.",
                "",
            ]
        )

        feature_cols = [
            c
            for c in df.columns
            if c
            not in {
                "RaceID",
                "season",
                "round",
                "race_name",
                "driver_id",
                "constructor_id",
                "finish_position",
                "is_dnf",
                "safety_car_flag",
            }
        ]
        nulos = df[feature_cols].isna().sum().sort_values(ascending=False)
        linhas.extend(["## Cobertura de features", ""])
        for coluna, total in nulos.items():
            status = "OK" if int(total) == 0 else f"{int(total)} nulos"
            linhas.append(f"- `{coluna}`: {status}")
        linhas.append("")

    if RELATORIO_2026.exists():
        linhas.extend(
            [
                "## Relatorio de origem",
                "",
                f"- `{RELATORIO_2026.relative_to(BASE_DIR)}`",
                "",
            ]
        )

    OUTPUT_2026.write_text("\n".join(linhas), encoding="utf-8")


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    resumo, folds = carregar_metricas()
    grafico_barras_metricas(resumo)
    grafico_folds(folds)
    importancias = carregar_importancias()
    dominantes = grafico_importancias(importancias)
    escrever_relatorio(resumo, dominantes)
    escrever_validacao_2026()
    print(f"Figuras geradas em: {FIGURES_DIR}")
    print(f"Relatorio: {OUTPUT_RELATORIO}")
    print(f"Validacao 2026: {OUTPUT_2026}")


if __name__ == "__main__":
    main()
