from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# Caminhos do projeto
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "processed" / "dataset_feature_engineering_ready_2018_2025.csv"

OUTPUT_DIR = BASE_DIR / "reports" / "eda_dataset_tratado" / "figures" / "eda_principal"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Colunas principais
TARGET_COL = "finish_position"
SEASON_COL = "season"
GRID_COL = "grid_position"
QUALI_COL = "qualifying_position"
PIT_COL = "fastf1_pit_in_count"
WEATHER_COL = "weather_impact_factor"
TRACK_COMPLEXITY_COL = "track_complexity"
AVG_PIT_COL = "avg_pit_stops_circuit"


def salvar_grafico(nome_arquivo: str) -> None:
    # salva o grafico em PNG
    caminho = OUTPUT_DIR / nome_arquivo
    plt.tight_layout()
    plt.savefig(caminho, dpi=300)
    plt.close()


def coluna_existe(df: pd.DataFrame, coluna: str) -> bool:
    # verifica se a coluna existe no df
    return coluna in df.columns


def expandir_colunas_outlier(df: pd.DataFrame) -> pd.Series:
    # expande outlier_colunas pra contar causas
    if "outlier_colunas" not in df.columns:
        return pd.Series(dtype="object")

    valores = []
    for item in df.loc[df.get("outlier_flag", 0) == 1, "outlier_colunas"].fillna(""):
        partes = [parte.strip() for parte in str(item).split(";") if parte.strip()]
        valores.extend(partes)

    return pd.Series(valores, dtype="object")


def gerar_resumo_outliers(df: pd.DataFrame) -> None:
    # gera painel com status, causas e localizacao dos outliers
    if "outlier_flag" not in df.columns:
        print("Coluna outlier_flag nao encontrada para o grafico 07.")
        return

    outliers = df[df["outlier_flag"] == 1].copy()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    if "outlier_tipo" in df.columns:
        tipo_counts = (
            outliers["outlier_tipo"]
            .value_counts()
            .reindex(["outlier_legitimo", "outlier_revisao", "outlier_espurio"], fill_value=0)
        )
        tipo_counts.plot(kind="bar", ax=axes[0, 0], color=["#2c7fb8", "#fdae61", "#d7191c"])
        for container in axes[0, 0].containers:
            axes[0, 0].bar_label(container, fmt="%d", padding=3)
        axes[0, 0].set_title("Classificacao dos outliers identificados")
        axes[0, 0].set_ylabel("Quantidade")
        axes[0, 0].set_xlabel("")
        axes[0, 0].tick_params(axis="x", rotation=25)
        axes[0, 0].set_ylim(0, max(1, int(tipo_counts.max() * 1.25)))
    else:
        flag_counts = outliers["outlier_flag"].value_counts().sort_index()
        flag_counts.plot(kind="bar", ax=axes[0, 0])
        for container in axes[0, 0].containers:
            axes[0, 0].bar_label(container, fmt="%d", padding=3)
        axes[0, 0].set_title("Outliers identificados")
        axes[0, 0].set_ylabel("Quantidade")

    colunas_outlier = expandir_colunas_outlier(df)
    if not colunas_outlier.empty:
        colunas_outlier.value_counts().sort_values().plot(kind="barh", ax=axes[0, 1])
        for container in axes[0, 1].containers:
            axes[0, 1].bar_label(container, fmt="%d", padding=3)
        axes[0, 1].set_title("Colunas que originaram outliers")
        axes[0, 1].set_xlabel("Quantidade")
    else:
        axes[0, 1].text(0.5, 0.5, "Sem colunas de outlier", ha="center", va="center")
        axes[0, 1].set_axis_off()

    if not outliers.empty and "season" in outliers.columns:
        outliers["season"].value_counts().sort_index().plot(kind="bar", ax=axes[1, 0])
        for container in axes[1, 0].containers:
            axes[1, 0].bar_label(container, fmt="%d", padding=3)
        axes[1, 0].set_title("Outliers por temporada")
        axes[1, 0].set_xlabel("Temporada")
        axes[1, 0].set_ylabel("Quantidade")
        axes[1, 0].tick_params(axis="x", rotation=0)
    else:
        axes[1, 0].text(0.5, 0.5, "Nenhum outlier identificado", ha="center", va="center")
        axes[1, 0].set_axis_off()

    axes[1, 1].set_axis_off()
    total = len(df)
    total_outliers = len(outliers)
    pct_outliers = 100 * total_outliers / total if total else 0
    revisao = int((outliers.get("outlier_tipo", pd.Series(dtype="object")) == "outlier_revisao").sum())
    legitimos = int((outliers.get("outlier_tipo", pd.Series(dtype="object")) == "outlier_legitimo").sum())
    espurios = int((outliers.get("outlier_tipo", pd.Series(dtype="object")) == "outlier_espurio").sum())
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
    salvar_grafico("07_outlier_resumo.png")


def gerar_distribuicoes_outliers(df: pd.DataFrame) -> None:
    # gera boxplots separados pra evitar escala misturada
    if "outlier_flag" not in df.columns:
        return

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

    if not metricas:
        print("Nenhuma metrica FastF1 encontrada para distribuicoes de outliers.")
        return

    n_cols = 4
    n_rows = (len(metricas) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3.8 * n_rows))
    axes = axes.flatten()

    for ax, col in zip(axes, metricas):
        normal = df.loc[df["outlier_flag"] == 0, col].dropna()
        outlier = df.loc[df["outlier_flag"] == 1, col].dropna()

        ax.boxplot([normal, outlier], tick_labels=["Normal", "Outlier"], showfliers=False)
        if not outlier.empty:
            ax.scatter(
                [2] * len(outlier),
                outlier,
                alpha=0.7,
                s=18,
                marker="o",
                label="Registros outlier",
            )
        ax.set_title(col)
        ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)

    for ax in axes[len(metricas):]:
        ax.set_axis_off()

    fig.suptitle("Distribuicao das metricas associadas aos outliers", fontsize=14)
    salvar_grafico("07b_outlier_distribuicoes_separadas.png")


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    print("Dataset carregado com sucesso.")
    print(f"Linhas: {df.shape[0]}")
    print(f"Colunas: {df.shape[1]}")

    # valores ausentes
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False).head(20)

    if not missing.empty:
        plt.figure(figsize=(10, 6))
        missing.sort_values().plot(kind="barh")
        plt.title("Principais colunas com valores ausentes")
        plt.xlabel("Quantidade de valores ausentes")
        plt.ylabel("Colunas")
        salvar_grafico("01_missing_values.png")
    else:
        plt.figure(figsize=(8, 4.5))
        plt.bar(["Dataset tratado"], [0], color="#2c7fb8")
        plt.title("Valores ausentes no dataset tratado")
        plt.ylabel("Quantidade de valores ausentes")
        plt.ylim(0, 1)
        plt.text(0, 0.08, "0 valores ausentes", ha="center", va="bottom", fontsize=12)
        salvar_grafico("01_missing_values.png")

    # distribuicao da variavel-alvo
    if coluna_existe(df, TARGET_COL):
        df_target = df.dropna(subset=[TARGET_COL]).copy()

        plt.figure(figsize=(8, 5))
        plt.hist(df_target[TARGET_COL], bins=20)
        plt.title("Distribuição da posição final")
        plt.xlabel("Posição final")
        plt.ylabel("Frequência")
        salvar_grafico("02_target_distribution.png")
    else:
        print(f"Coluna não encontrada: {TARGET_COL}")

    # média da posição final por temporada
    if coluna_existe(df, TARGET_COL) and coluna_existe(df, SEASON_COL):
        df_season = df.dropna(subset=[TARGET_COL, SEASON_COL]).copy()

        target_by_season = (
            df_season
            .groupby(SEASON_COL)[TARGET_COL]
            .mean()
            .sort_index()
        )

        plt.figure(figsize=(9, 5))
        target_by_season.plot(kind="line", marker="o")
        plt.title("Média da posição final por temporada")
        plt.xlabel("Temporada")
        plt.ylabel("Média da posição final")
        plt.grid(True, linewidth=0.5)
        salvar_grafico("03_target_by_season.png")
    else:
        print("Colunas necessárias não encontradas para o gráfico 03.")

    # grid position x finish position
    if coluna_existe(df, GRID_COL) and coluna_existe(df, TARGET_COL):
        df_grid = df.dropna(subset=[GRID_COL, TARGET_COL]).copy()

        plt.figure(figsize=(7, 5))
        plt.scatter(df_grid[GRID_COL], df_grid[TARGET_COL], alpha=0.5)
        plt.title("Relação entre posição de largada e posição final")
        plt.xlabel("Posição de largada")
        plt.ylabel("Posição final")
        plt.grid(True, linewidth=0.5)
        salvar_grafico("04_grid_vs_finish.png")
    else:
        print("Colunas necessárias não encontradas para o gráfico 04.")

    # qualifying position x finish position
    if coluna_existe(df, QUALI_COL) and coluna_existe(df, TARGET_COL):
        df_quali = df.dropna(subset=[QUALI_COL, TARGET_COL]).copy()

        plt.figure(figsize=(7, 5))
        plt.scatter(df_quali[QUALI_COL], df_quali[TARGET_COL], alpha=0.5)
        plt.title("Relação entre classificação e posição final")
        plt.xlabel("Posição na classificação")
        plt.ylabel("Posição final")
        plt.grid(True, linewidth=0.5)
        salvar_grafico("05_qualifying_vs_finish.png")
    else:
        print(f"Coluna não encontrada ou ausente: {QUALI_COL}")

    # matriz de correlação das principais variáveis
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    cols_corr_prioritarias = [
        TARGET_COL,
        GRID_COL,
        QUALI_COL,
        PIT_COL,
        "points",
        "laps",
        "fastf1_avg_lap_time",
        "fastf1_best_lap_time",
        "fastf1_avg_sector1",
        "fastf1_avg_sector2",
        "fastf1_avg_sector3",
        "fastf1_max_tyre_life",
        "fastf1_stints_count",
    ]

    cols_corr = [col for col in cols_corr_prioritarias if col in numeric_cols]

    if len(cols_corr) >= 2:
        corr = df[cols_corr].corr()

        plt.figure(figsize=(10, 8))
        plt.imshow(corr, aspect="auto")
        plt.colorbar(label="Correlação")
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.columns)), corr.columns)
        plt.title("Matriz de correlação das principais variáveis numéricas")
        salvar_grafico("06_correlation_heatmap.png")
    else:
        print("Não há colunas numéricas suficientes para o gráfico 06.")

    # outliers interpretáveis
    gerar_resumo_outliers(df)
    gerar_distribuicoes_outliers(df)

    # distribuicao de pit stops
    if coluna_existe(df, PIT_COL):
        df_pit = df.dropna(subset=[PIT_COL]).copy()

        plt.figure(figsize=(8, 5))
        plt.hist(df_pit[PIT_COL], bins=10)
        plt.title("Distribuição da quantidade de pit stops")
        plt.xlabel("Quantidade de pit stops")
        plt.ylabel("Frequência")
        salvar_grafico("08_pit_stops_distribution.png")
    else:
        print(f"Coluna não encontrada: {PIT_COL}")

    # ganho ou perda de posicoes
    if coluna_existe(df, GRID_COL) and coluna_existe(df, TARGET_COL):
        df_gain = df.dropna(subset=[GRID_COL, TARGET_COL]).copy()
        df_gain["position_gain"] = df_gain[GRID_COL] - df_gain[TARGET_COL]

        plt.figure(figsize=(8, 5))
        plt.hist(df_gain["position_gain"], bins=25)
        plt.axvline(0, linestyle="--", linewidth=1)
        plt.title("Distribuição de ganho ou perda de posições")
        plt.xlabel("Ganho de posições")
        plt.ylabel("Frequência")
        salvar_grafico("09_position_gain_distribution.png")
    else:
        print("Colunas necessárias não encontradas para o gráfico 09.")

    # distribuicao do impacto climatico
    if coluna_existe(df, WEATHER_COL):
        plt.figure(figsize=(8, 5))
        plt.hist(df[WEATHER_COL].dropna(), bins=25)
        plt.title("Distribuição do fator de impacto climático")
        plt.xlabel("Fator de impacto climático")
        plt.ylabel("Frequência")
        salvar_grafico("10_weather_impact_distribution.png")
    else:
        print(f"Coluna não encontrada: {WEATHER_COL}")

    # distribuição da complexidade de pista
    if coluna_existe(df, TRACK_COMPLEXITY_COL):
        plt.figure(figsize=(8, 5))
        plt.hist(df[TRACK_COMPLEXITY_COL].dropna(), bins=25)
        plt.title("Distribuição da complexidade de pista")
        plt.xlabel("Complexidade de pista")
        plt.ylabel("Frequência")
        salvar_grafico("11_track_complexity_distribution.png")
    else:
        print(f"Coluna não encontrada: {TRACK_COMPLEXITY_COL}")

    # resumo temporal de métricas principais
    metricas_temporais = [
        TARGET_COL,
        WEATHER_COL,
        TRACK_COMPLEXITY_COL,
        AVG_PIT_COL,
    ]
    metricas_temporais = [col for col in metricas_temporais if col in df.columns]

    if coluna_existe(df, SEASON_COL) and metricas_temporais:
        resumo_temporal = df.groupby(SEASON_COL)[metricas_temporais].mean().sort_index()

        plt.figure(figsize=(10, 5))
        for col in metricas_temporais:
            plt.plot(resumo_temporal.index, resumo_temporal[col], marker="o", label=col)
        plt.title("Evolução temporal das métricas principais")
        plt.xlabel("Temporada")
        plt.ylabel("Média")
        plt.grid(True, linewidth=0.5)
        plt.legend()
        salvar_grafico("12_temporal_summary.png")
    else:
        print("Colunas necessárias não encontradas para o gráfico 12.")

    print("Gráficos principais gerados com sucesso em:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
