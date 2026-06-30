from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from metricas import calcular_metricas
from tuning_utils import calcular_sample_weight, carregar_decay_escolhido, score_composto_metricas


# caminhos base do projeto
BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports" / "modelagem"
FIGURES_DIR = REPORTS_DIR / "figures" / "semana3_2026"

# onde estao os dados de treino e os dados 2026
INPUT_X = PROCESSED_DIR / "dataset_modelagem_X_2018_2025.csv"
INPUT_Y = PROCESSED_DIR / "dataset_modelagem_y_2018_2025.csv"
INPUT_2026 = PROCESSED_DIR / "fastf1_2026_available.csv"

# onde ficam os params de cada modelo (salvos pelo optuna)
PARAMS = {
    "ridge_baseline": REPORTS_DIR / "ridge_best_params.json",
    "lightgbm_tuned": REPORTS_DIR / "optuna_lightgbm_best_params.json",
    "random_forest_tuned": REPORTS_DIR / "optuna_randomforest_best_params.json",
    "xgboost_tuned": REPORTS_DIR / "optuna_xgboost_best_params.json",
}

# arquivos de saída
OUTPUT_PREDICOES = REPORTS_DIR / "predicoes_2026_semana3.csv"
OUTPUT_METRICAS_CORRIDA = REPORTS_DIR / "metricas_2026_por_corrida.csv"
OUTPUT_METRICAS_RESUMO = REPORTS_DIR / "metricas_2026_resumo.csv"
OUTPUT_ERRO_GRID = REPORTS_DIR / "analise_erro_2026_por_grid.csv"

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


# wrapper pro Ridge que já inclui o scaler junto
class RidgeBaselineRegressor:
    def __init__(self, alpha: float) -> None:
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha, random_state=42)

    def fit(self, x, y, sample_weight=None):
        x_scaled = self.scaler.fit_transform(x)
        self.model.fit(x_scaled, y, sample_weight=sample_weight)
        return self

    def predict(self, x):
        return self.model.predict(self.scaler.transform(x))


def carregar_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def filtrar_params_modelo(modelo: str, params: dict) -> dict:
    # joga fora as chaves de metadado que não são parâmetros do modelo
    ignorar = {
        "score_composto_tuning",
        "tempo_tuning_segundos",
        "lightgbm_version",
        "modelo",
        "normalizacao",
    }
    return {k: v for k, v in params.items() if k not in ignorar}


def criar_modelo(modelo: str):
    # carrega os params do json e instancia o modelo certo
    params = filtrar_params_modelo(modelo, carregar_json(PARAMS[modelo]))

    if modelo == "ridge_baseline":
        return RidgeBaselineRegressor(alpha=float(params["alpha"]))

    if modelo == "lightgbm_tuned":
        return lgb.LGBMRegressor(
            objective="regression",
            random_state=42,
            n_jobs=4,
            verbosity=-1,
            **params,
        )

    if modelo == "random_forest_tuned":
        return RandomForestRegressor(
            random_state=42,
            n_jobs=4,
            bootstrap=True,
            **params,
        )

    if modelo == "xgboost_tuned":
        return XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=4,
            **params,
        )

    raise ValueError(f"Modelo desconhecido: {modelo}")


def carregar_dados() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    x_train = pd.read_csv(INPUT_X)
    y_train = pd.read_csv(INPUT_Y)
    df_2026_raw = pd.read_csv(INPUT_2026)

    # conta quantas linhas não têm resultado de corrida ainda
    n_finish_nan = int(df_2026_raw["finish_position"].isna().sum())
    df_2026 = df_2026_raw.dropna(subset=["finish_position"]).copy()

    # verifica se o dataset 2026 tem todas as features que o modelo espera
    faltantes = sorted(set(x_train.columns).difference(df_2026.columns))
    if faltantes:
        raise ValueError(f"Features finais ausentes na base 2026: {faltantes}")

    x_2026 = df_2026[x_train.columns].copy()
    return x_train, y_train, df_2026, x_2026, n_finish_nan


def prever_2026() -> tuple[pd.DataFrame, int]:
    x_train, y_train, df_2026, x_2026, n_finish_nan = carregar_dados()

    # pega o decay configurado e calcula os pesos de treino
    decay = carregar_decay_escolhido()
    sample_weight = calcular_sample_weight(
        y_train=y_train,
        valid_season=2026,
        decay=decay,
    )

    predicoes = []
    base_cols = [
        "RaceID",
        "season",
        "round",
        "race_name",
        "driver_id",
        "constructor_id",
        "finish_position",
        "qualifying_position",
    ]

    # treina cada modelo no histórico e aplica nos dados 2026
    for modelo in MODEL_LABELS:
        estimador = criar_modelo(modelo)
        estimador.fit(x_train, y_train["finish_position"], sample_weight=sample_weight)

        df_pred = df_2026[base_cols].copy()
        df_pred["modelo"] = modelo
        df_pred["modelo_label"] = MODEL_LABELS[modelo]
        df_pred["pred_finish_position"] = estimador.predict(x_2026)
        df_pred["erro"] = df_pred["pred_finish_position"] - df_pred["finish_position"]
        df_pred["erro_abs"] = df_pred["erro"].abs()
        predicoes.append(df_pred)

    return pd.concat(predicoes, ignore_index=True), n_finish_nan


def calcular_metricas_2026(predicoes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metricas_corrida = []

    # calcula as métricas separado por modelo e por corrida
    for (modelo, modelo_label, round_num, race_name), grupo in predicoes.groupby(
        ["modelo", "modelo_label", "round", "race_name"]
    ):
        metricas = calcular_metricas(grupo)
        metricas_corrida.append(
            {
                "modelo": modelo,
                "modelo_label": modelo_label,
                "round": round_num,
                "race_name": race_name,
                "n_valid": len(grupo),
                **metricas,
            }
        )

    df_corrida = pd.DataFrame(metricas_corrida)

    # agrega por modelo pra ter o resumo geral
    resumo = (
        df_corrida.groupby(["modelo", "modelo_label"], as_index=False)
        .agg(
            mae_medio=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_medio=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            r2_medio=("r2", "mean"),
            kendall_tau_medio=("kendall_tau", "mean"),
            n_corridas=("round", "nunique"),
            n_predicoes=("n_valid", "sum"),
        )
    )

    # score composto igual ao critério oficial do TCC
    resumo["score_composto_exploratorio"] = resumo["modelo"].map(
        {
            modelo: score_composto_metricas(grupo)
            for modelo, grupo in df_corrida.groupby("modelo")
        }
    )
    resumo = resumo.sort_values(
        ["score_composto_exploratorio", "mae_medio"],
        ascending=[False, True],
    ).reset_index(drop=True)
    return df_corrida, resumo


def analisar_erro_grid(predicoes: pd.DataFrame) -> pd.DataFrame:
    df = predicoes.copy()

    # divide os pilotos em faixas de grid pra ver onde o modelo erra mais
    df["faixa_grid"] = pd.cut(
        df["qualifying_position"],
        bins=[0, 5, 10, 15, 25],
        labels=["P1-P5", "P6-P10", "P11-P15", "P16+"],
        include_lowest=True,
    )
    return (
        df.groupby(["modelo", "modelo_label", "faixa_grid"], observed=True)
        .agg(
            mae=("erro_abs", "mean"),
            erro_medio=("erro", "mean"),
            n=("erro_abs", "size"),
        )
        .reset_index()
    )


def _savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def gerar_graficos(
    metricas_corrida: pd.DataFrame,
    resumo: pd.DataFrame,
    erro_grid: pd.DataFrame,
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # gráfico 1: MAE médio geral por modelo
    plt.figure(figsize=(8.5, 4.8))
    ax = sns.barplot(
        data=resumo,
        x="modelo_label",
        y="mae_medio",
        order=MODEL_ORDER,
        hue="modelo_label",
        palette=PALETTE,
        dodge=False,
        legend=False,
    )
    ax.set_title("MAE exploratorio em 2026")
    ax.set_xlabel("")
    ax.set_ylabel("MAE medio por corrida")
    ax.bar_label(ax.containers[0], fmt="%.3f", padding=3, fontsize=9)
    _savefig(FIGURES_DIR / "01_mae_2026_resumo.png")

    # gráfico 2: MAE ao longo das corridas
    plt.figure(figsize=(9, 5))
    ax = sns.lineplot(
        data=metricas_corrida,
        x="round",
        y="mae",
        hue="modelo_label",
        hue_order=MODEL_ORDER,
        marker="o",
        palette=PALETTE,
        linewidth=2,
    )
    ax.set_title("MAE por corrida disponivel de 2026")
    ax.set_xlabel("Round")
    ax.set_ylabel("MAE")
    ax.legend(title="Modelo", frameon=False)
    _savefig(FIGURES_DIR / "02_mae_2026_por_corrida.png")

    # gráfico 3: kendall tau por corrida
    plt.figure(figsize=(9, 5))
    ax = sns.lineplot(
        data=metricas_corrida,
        x="round",
        y="kendall_tau",
        hue="modelo_label",
        hue_order=MODEL_ORDER,
        marker="o",
        palette=PALETTE,
        linewidth=2,
    )
    ax.set_title("Kendall tau por corrida disponivel de 2026")
    ax.set_xlabel("Round")
    ax.set_ylabel("Kendall tau")
    ax.legend(title="Modelo", frameon=False)
    _savefig(FIGURES_DIR / "03_kendall_2026_por_corrida.png")

    # gráfico 4: erro por faixa de grid - mostra se erra mais em pilotos de fundo
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(
        data=erro_grid,
        x="faixa_grid",
        y="mae",
        hue="modelo_label",
        hue_order=MODEL_ORDER,
        palette=PALETTE,
    )
    ax.set_title("Erro medio absoluto por faixa de grid - 2026")
    ax.set_xlabel("Faixa de qualifying/grid")
    ax.set_ylabel("MAE")
    ax.legend(title="Modelo", frameon=False)
    _savefig(FIGURES_DIR / "04_erro_2026_por_grid.png")


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    predicoes, n_finish_nan = prever_2026()
    metricas_corrida, resumo = calcular_metricas_2026(predicoes)
    erro_grid = analisar_erro_grid(predicoes)

    # salva os csvs de resultado
    predicoes.to_csv(OUTPUT_PREDICOES, index=False, encoding="utf-8-sig")
    metricas_corrida.to_csv(OUTPUT_METRICAS_CORRIDA, index=False, encoding="utf-8-sig")
    resumo.to_csv(OUTPUT_METRICAS_RESUMO, index=False, encoding="utf-8-sig")
    erro_grid.to_csv(OUTPUT_ERRO_GRID, index=False, encoding="utf-8-sig")

    gerar_graficos(metricas_corrida, resumo, erro_grid)

    print("Analise exploratoria 2026 concluida.")
    print(resumo.to_string(index=False))


if __name__ == "__main__":
    main()
