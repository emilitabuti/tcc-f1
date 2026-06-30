from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# caminhos das pastas principais
BASE_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = BASE_DIR / "reports" / "modelagem"
FIGURES_DIR = REPORTS_DIR / "figures" / "semana3"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# csvs com as metricas dos modelos tunados
METRICAS_RESUMO = REPORTS_DIR / "tabela_metricas_tunadas_4modelos_resumo.csv"
METRICAS_FOLDS = REPORTS_DIR / "tabela_metricas_tunadas_4modelos.csv"
FEATURE_IMPORTANCE = {
    "LightGBM": REPORTS_DIR / "feature_importance_lgb.csv",
    "Random Forest": REPORTS_DIR / "feature_importance_rf.csv",
    "XGBoost": REPORTS_DIR / "feature_importance_xgb.csv",
}
FASTF1_2026 = PROCESSED_DIR / "fastf1_2026_available.csv"
RELATORIO_2026 = PROCESSED_DIR / "relatorio_update_2026.txt"

OUTPUT_FEATURES_DOMINANTES = REPORTS_DIR / "tabela_features_dominantes_semana3.csv"

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
    # mapeia o nome interno do modelo pro label bonitinho
    resumo["modelo_label"] = resumo["modelo"].map(MODEL_LABELS)
    folds["modelo_label"] = folds["modelo"].map(MODEL_LABELS)
    return resumo, folds


def grafico_barras_metricas(resumo: pd.DataFrame) -> None:
    # lista de metricas que vao virar grafico de barra
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
        # adiciona barra de erro só pro MAE e RMSE
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
    # evolução das métricas ao longo dos folds temporais
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
    # junta as importancias dos tres modelos de arvore num dataframe so
    frames = []
    for modelo, path in FEATURE_IMPORTANCE.items():
        df = _read_csv(path)
        df["modelo_label"] = modelo
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def grafico_importancias(importancias: pd.DataFrame) -> pd.DataFrame:
    dominantes = []
    for modelo in ["LightGBM", "Random Forest", "XGBoost"]:
        # pega só as top 8 features de cada modelo
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
    # salva so as colunas relevantes
    df_dominantes[
        ["modelo_label", "rank", "feature", "importance_type", "importance_norm_media"]
    ].to_csv(OUTPUT_FEATURES_DOMINANTES, index=False, encoding="utf-8-sig")
    return df_dominantes


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    resumo, folds = carregar_metricas()
    grafico_barras_metricas(resumo)
    grafico_folds(folds)
    importancias = carregar_importancias()
    grafico_importancias(importancias)
    print(f"Figuras geradas em: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
