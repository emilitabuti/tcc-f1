from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "processed" / "dataset_feature_engineering_ready_2018_2025.csv"
OUTPUT_DIR = BASE_DIR / "reports" / "eda_dataset_tratado" / "figures" / "eda_principal"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def salvar(nome_arquivo: str) -> None:
    caminho = OUTPUT_DIR / nome_arquivo
    plt.tight_layout()
    plt.savefig(caminho, dpi=300)
    plt.close()


def expandir_colunas_outlier(outliers: pd.DataFrame) -> pd.Series:
    valores = []
    for item in outliers.get("outlier_colunas", pd.Series(dtype="object")).fillna(""):
        partes = [parte.strip() for parte in str(item).split(";") if parte.strip()]
        valores.extend(partes)
    return pd.Series(valores, dtype="object")


def gerar_resumo(df: pd.DataFrame) -> None:
    outliers = df[df["outlier_flag"] == 1].copy()

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    tipo_counts = (
        outliers["outlier_tipo"]
        .value_counts()
        .reindex(["outlier_legitimo", "outlier_revisao", "outlier_espurio"], fill_value=0)
    )
    tipo_counts.plot(kind="bar", ax=axes[0, 0], color=["#2c7fb8", "#fdae61", "#d7191c"])
    axes[0, 0].set_title("Classificacao dos outliers identificados")
    axes[0, 0].set_ylabel("Quantidade")
    axes[0, 0].set_xlabel("")
    axes[0, 0].tick_params(axis="x", rotation=20)
    axes[0, 0].set_ylim(0, max(1, int(tipo_counts.max() * 1.25)))
    for container in axes[0, 0].containers:
        axes[0, 0].bar_label(container, fmt="%d", padding=3)

    causas = expandir_colunas_outlier(outliers)
    if causas.empty:
        axes[0, 1].text(0.5, 0.5, "Sem outliers identificados", ha="center", va="center")
        axes[0, 1].set_axis_off()
    else:
        causas.value_counts().sort_values().plot(kind="barh", ax=axes[0, 1], color="#2c7fb8")
        axes[0, 1].set_title("Colunas que originaram outliers")
        axes[0, 1].set_xlabel("Quantidade")
        for container in axes[0, 1].containers:
            axes[0, 1].bar_label(container, fmt="%d", padding=3)

    if outliers.empty:
        axes[1, 0].text(0.5, 0.5, "Sem outliers identificados", ha="center", va="center")
        axes[1, 0].set_axis_off()
    else:
        outliers["season"].value_counts().sort_index().plot(kind="bar", ax=axes[1, 0], color="#2c7fb8")
        axes[1, 0].set_title("Outliers por temporada")
        axes[1, 0].set_xlabel("Temporada")
        axes[1, 0].set_ylabel("Quantidade")
        axes[1, 0].tick_params(axis="x", rotation=0)
        for container in axes[1, 0].containers:
            axes[1, 0].bar_label(container, fmt="%d", padding=3)

    axes[1, 1].set_axis_off()
    total = len(df)
    total_outliers = len(outliers)
    pct_outliers = 100 * total_outliers / total if total else 0
    legitimos = int((outliers["outlier_tipo"] == "outlier_legitimo").sum())
    revisao = int((outliers["outlier_tipo"] == "outlier_revisao").sum())
    espurios = int((outliers["outlier_tipo"] == "outlier_espurio").sum())

    texto = (
        f"Total de registros: {total}\n"
        f"Outliers identificados: {total_outliers} ({pct_outliers:.2f}%)\n"
        f"Outliers legitimos: {legitimos}\n"
        f"Outliers em revisao: {revisao}\n"
        f"Outliers espurios: {espurios}\n\n"
        "Leitura metodologica:\n"
        "- A proporcao de outliers e baixa.\n"
        "- Todos os outliers estao classificados.\n"
        "- As causas se concentram em metricas FastF1 de tempo/setor."
    )
    axes[1, 1].text(0.02, 0.95, texto, va="top", fontsize=11)

    fig.suptitle("Resumo interpretavel dos outliers", fontsize=14)
    salvar("07_outlier_resumo.png")


def gerar_distribuicoes(df: pd.DataFrame) -> None:
    metricas = [
        "fastf1_avg_lap_time",
        "fastf1_best_lap_time",
        "fastf1_avg_sector1",
        "fastf1_avg_sector2",
        "fastf1_avg_sector3",
        "fastf1_max_tyre_life",
        "fastf1_stints_count",
        "fastf1_pit_in_count",
    ]
    metricas = [col for col in metricas if col in df.columns]
    outlier_mask = df["outlier_flag"] == 1

    fig, axes = plt.subplots(2, 4, figsize=(15, 7.6))
    axes = axes.flatten()

    for ax, col in zip(axes, metricas):
        normal = df.loc[~outlier_mask, col].dropna()
        outlier = df.loc[outlier_mask, col].dropna()
        ax.boxplot([normal, outlier], tick_labels=["Normal", "Outlier"], showfliers=False)
        ax.scatter([2] * len(outlier), outlier, alpha=0.7, s=18)
        ax.set_title(col)
        ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)

    for ax in axes[len(metricas):]:
        ax.set_axis_off()

    fig.suptitle("Distribuicao das metricas associadas aos outliers", fontsize=14)
    salvar("07b_outlier_distribuicoes_separadas.png")


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    gerar_resumo(df)
    gerar_distribuicoes(df)
    print(f"Graficos de outliers gerados em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
