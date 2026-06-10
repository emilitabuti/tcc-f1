from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
REPORTS_DIR = ROOT_DIR / "reports" / "modelagem"

sys.path.insert(0, str(SRC_DIR))

from metricas import calcular_metricas  # noqa: E402


INPUT_X = PROCESSED_DIR / "dataset_modelagem_X_2018_2025.csv"
INPUT_Y = PROCESSED_DIR / "dataset_modelagem_y_2018_2025.csv"
INPUT_PARAMS = REPORTS_DIR / "optuna_xgboost_best_params.json"
INPUT_DECAY = REPORTS_DIR / "time_decay_escolhido_xgboost.txt"

OUTPUT_METRICAS = REPORTS_DIR / "experimento_dnf_rates_metricas.csv"
OUTPUT_RELATORIO = REPORTS_DIR / "experimento_dnf_rates_relatorio.md"

DECAY_PADRAO = 0.95
FOLDS = [
    {"train_until": 2022, "valid_season": 2023},
    {"train_until": 2023, "valid_season": 2024},
    {"train_until": 2024, "valid_season": 2025},
]

FEATURES_DNF = ["driver_dnf_rate", "constructor_dnf_rate"]
EXPERIMENTOS = [
    {
        "experimento": "A",
        "cenario": "base_15_features",
        "remover_features": [],
        "descricao": "Dataset final atual, com driver_dnf_rate e constructor_dnf_rate.",
    },
    {
        "experimento": "B",
        "cenario": "sem_driver_dnf_rate",
        "remover_features": ["driver_dnf_rate"],
        "descricao": "Remove apenas a taxa historica de DNF atribuida ao piloto.",
    },
    {
        "experimento": "C",
        "cenario": "sem_constructor_dnf_rate",
        "remover_features": ["constructor_dnf_rate"],
        "descricao": "Remove apenas a taxa historica de DNF mecanico do construtor.",
    },
    {
        "experimento": "D",
        "cenario": "sem_duas_dnf_rates",
        "remover_features": FEATURES_DNF,
        "descricao": "Remove simultaneamente as duas taxas de DNF.",
    },
]


def carregar_dados() -> tuple[pd.DataFrame, pd.DataFrame]:
    x = pd.read_csv(INPUT_X)
    y = pd.read_csv(INPUT_Y)

    if len(x) != len(y):
        raise RuntimeError(f"X e y com tamanhos diferentes: {len(x)} vs {len(y)}")

    faltantes = [feature for feature in FEATURES_DNF if feature not in x.columns]
    if faltantes:
        raise RuntimeError(f"Features DNF ausentes em X: {faltantes}")

    if x.isna().sum().sum() > 0 or y.isna().sum().sum() > 0:
        raise RuntimeError("X ou y contem valores nulos.")

    return x, y


def carregar_parametros_xgboost() -> dict:
    if not INPUT_PARAMS.exists():
        raise FileNotFoundError(f"Parametros XGBoost ausentes: {INPUT_PARAMS}")

    params = json.loads(INPUT_PARAMS.read_text(encoding="utf-8"))
    params.pop("tempo_tuning_segundos", None)
    return params


def carregar_decay() -> float:
    if not INPUT_DECAY.exists():
        return DECAY_PADRAO

    for linha in INPUT_DECAY.read_text(encoding="utf-8").splitlines():
        if linha.lower().startswith("time-decay escolhido:"):
            return float(linha.split(":", 1)[1].strip())

    return DECAY_PADRAO


def criar_modelo(params: dict) -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=4,
        **params,
    )


def calcular_sample_weight(y_train: pd.DataFrame, valid_season: int, decay: float) -> np.ndarray:
    distancia = valid_season - y_train["season"]
    distancia = distancia.clip(lower=0)
    return np.power(decay, distancia).to_numpy()


def rodar_fold(
    x: pd.DataFrame,
    y: pd.DataFrame,
    params: dict,
    decay: float,
    features: list[str],
    experimento: str,
    cenario: str,
    train_until: int,
    valid_season: int,
) -> dict:
    train_mask = y["season"] <= train_until
    valid_mask = y["season"] == valid_season

    if train_mask.sum() == 0:
        raise RuntimeError(f"Nenhuma linha de treino ate {train_until}.")
    if valid_mask.sum() == 0:
        raise RuntimeError(f"Nenhuma linha de validacao para {valid_season}.")

    sample_weight = calcular_sample_weight(
        y_train=y.loc[train_mask],
        valid_season=valid_season,
        decay=decay,
    )

    modelo = criar_modelo(params)
    modelo.fit(
        x.loc[train_mask, features],
        y.loc[train_mask, "finish_position"],
        sample_weight=sample_weight,
    )

    df_pred = y.loc[valid_mask].copy()
    df_pred["pred_finish_position"] = modelo.predict(x.loc[valid_mask, features])

    metricas = calcular_metricas(df_pred)
    metricas.update(
        {
            "experimento": experimento,
            "cenario": cenario,
            "train_until": train_until,
            "valid_season": valid_season,
            "decay": decay,
            "n_features": len(features),
            "n_train": int(train_mask.sum()),
            "n_valid": int(valid_mask.sum()),
        }
    )

    return metricas


def rodar_experimentos(x: pd.DataFrame, y: pd.DataFrame, params: dict, decay: float) -> pd.DataFrame:
    resultados = []
    features_base = list(x.columns)

    for cfg in EXPERIMENTOS:
        remover = set(cfg["remover_features"])
        features = [feature for feature in features_base if feature not in remover]

        for fold in FOLDS:
            resultados.append(
                rodar_fold(
                    x=x,
                    y=y,
                    params=params,
                    decay=decay,
                    features=features,
                    experimento=cfg["experimento"],
                    cenario=cfg["cenario"],
                    train_until=fold["train_until"],
                    valid_season=fold["valid_season"],
                )
            )

    df_metricas = pd.DataFrame(resultados)

    base_media = (
        df_metricas[df_metricas["experimento"] == "A"]
        .groupby("experimento", as_index=False)["mae"]
        .mean()
        .iloc[0]["mae"]
    )
    medias = (
        df_metricas
        .groupby(["experimento", "cenario"], as_index=False)
        .agg(
            mae_medio=("mae", "mean"),
            rmse_medio=("rmse", "mean"),
            r2_medio=("r2", "mean"),
            kendall_tau_medio=("kendall_tau", "mean"),
            top3_accuracy_medio=("top3_accuracy", "mean"),
        )
    )
    medias["delta_mae_medio_vs_base"] = medias["mae_medio"] - base_media

    return df_metricas.merge(medias, on=["experimento", "cenario"], how="left")


def formatar_markdown(df: pd.DataFrame) -> str:
    colunas = df.columns.tolist()
    linhas = [
        "| " + " | ".join(colunas) + " |",
        "| " + " | ".join(["---"] * len(colunas)) + " |",
    ]

    for _, row in df.iterrows():
        valores = []
        for col in colunas:
            valor = row[col]
            if isinstance(valor, float):
                valores.append(f"{valor:.6f}")
            else:
                valores.append(str(valor))
        linhas.append("| " + " | ".join(valores) + " |")

    return "\n".join(linhas)


def gerar_relatorio(df_metricas: pd.DataFrame, params: dict, decay: float) -> None:
    resumo = (
        df_metricas[
            [
                "experimento",
                "cenario",
                "n_features",
                "mae_medio",
                "delta_mae_medio_vs_base",
                "rmse_medio",
                "r2_medio",
                "kendall_tau_medio",
                "top3_accuracy_medio",
            ]
        ]
        .drop_duplicates()
        .sort_values("mae_medio")
        .reset_index(drop=True)
    )

    por_fold = df_metricas[
        [
            "experimento",
            "cenario",
            "valid_season",
            "n_features",
            "mae",
            "rmse",
            "r2",
            "kendall_tau",
            "top3_accuracy",
        ]
    ].sort_values(["experimento", "valid_season"])

    melhor = resumo.iloc[0]
    base = resumo[resumo["experimento"] == "A"].iloc[0]

    linhas = [
        "# Experimento DNF Rates - Ablacao de Features",
        "",
        "## Objetivo",
        "",
        (
            "Avaliar se as features `driver_dnf_rate` e `constructor_dnf_rate` "
            "melhoram a predicao de `finish_position` no dataset final DNF Excluded. "
            "O teste remove uma ou ambas as features e compara as metricas walk-forward "
            "contra a configuracao atual."
        ),
        "",
        "## Configuracao",
        "",
        f"- Modelo: XGBoost com hiperparametros ja tunados em `{INPUT_PARAMS}`.",
        f"- X: `{INPUT_X}`.",
        f"- y/metadados: `{INPUT_Y}`.",
        f"- Time-decay: `{decay}`.",
        "- Folds: treino ate 2022 -> 2023, treino ate 2023 -> 2024, treino ate 2024 -> 2025.",
        "- Target: `finish_position`.",
        "- Metricas: MAE, RMSE, R2, Kendall tau medio por corrida e Top-3 accuracy.",
        "",
        "## Cenarios",
        "",
    ]

    for cfg in EXPERIMENTOS:
        linhas.append(f"- {cfg['experimento']} - `{cfg['cenario']}`: {cfg['descricao']}")

    linhas.extend(
        [
            "",
            "## Parametros do Modelo",
            "",
            "```json",
            json.dumps(params, indent=2, sort_keys=True),
            "```",
            "",
            "## Resumo Medio",
            "",
            formatar_markdown(resumo),
            "",
            "## Resultados por Fold",
            "",
            formatar_markdown(por_fold),
            "",
            "## Interpretacao",
            "",
            (
                f"O melhor MAE medio foi do cenario {melhor['experimento']} "
                f"(`{melhor['cenario']}`), com MAE medio={melhor['mae_medio']:.6f}. "
                f"A configuracao base teve MAE medio={base['mae_medio']:.6f}."
            ),
            (
                "A diferenca e pequena e deve ser lida como evidencia de sinal fraco, "
                "nao como prova definitiva de que as taxas DNF devam ser removidas. "
                "Como o dataset de modelagem exclui DNFs, essas taxas historicas tendem "
                "a explicar pouco a posicao final entre pilotos que terminaram/classificaram "
                "a corrida."
            ),
            (
                "Metodologicamente, manter as features continua defensavel porque elas sao "
                "causais e baseadas no historico classificado completo de DNF. Para desempenho "
                "puro, porem, o teste sugere que `driver_dnf_rate` e uma candidata razoavel "
                "a ablação em versoes futuras do modelo."
            ),
            "",
            "## Artefatos Gerados",
            "",
            f"- `{OUTPUT_METRICAS}`",
            f"- `{OUTPUT_RELATORIO}`",
            "",
        ]
    )

    OUTPUT_RELATORIO.write_text("\n".join(linhas), encoding="utf-8")


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    x, y = carregar_dados()
    params = carregar_parametros_xgboost()
    decay = carregar_decay()

    df_metricas = rodar_experimentos(x, y, params, decay)
    df_metricas.to_csv(OUTPUT_METRICAS, index=False, encoding="utf-8-sig")
    gerar_relatorio(df_metricas, params, decay)

    print("Experimento de ablação DNF rates concluido.")
    print(
        df_metricas[
            [
                "experimento",
                "cenario",
                "valid_season",
                "n_features",
                "mae",
                "mae_medio",
                "delta_mae_medio_vs_base",
                "kendall_tau",
                "top3_accuracy",
            ]
        ].to_string(index=False)
    )
    print(f"\nArquivos gerados:\n- {OUTPUT_METRICAS}\n- {OUTPUT_RELATORIO}")


if __name__ == "__main__":
    main()
