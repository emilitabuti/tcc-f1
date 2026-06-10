from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from metricas import calcular_metricas


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports" / "modelagem"

INPUT_X = PROCESSED_DIR / "dataset_modelagem_X_2018_2025.csv"
INPUT_Y = PROCESSED_DIR / "dataset_modelagem_y_2018_2025.csv"
INPUT_PARAMS = REPORTS_DIR / "optuna_lightgbm_best_params.json"
INPUT_DECAY = REPORTS_DIR / "time_decay_escolhido_xgboost.txt"

OUTPUT_METRICAS = REPORTS_DIR / "experimento_janela_treino_2025_metricas.csv"
OUTPUT_PREDICOES = REPORTS_DIR / "experimento_janela_treino_2025_predicoes.csv"
OUTPUT_RELATORIO = REPORTS_DIR / "experimento_janela_treino_2025_relatorio.md"

VALID_SEASON = 2025
TRAIN_END = 2024
DECAY_PADRAO = 0.95

EXPERIMENTOS = [
    {"experimento": "A", "train_start": 2018, "train_end": TRAIN_END},
    {"experimento": "B", "train_start": 2019, "train_end": TRAIN_END},
    {"experimento": "C", "train_start": 2020, "train_end": TRAIN_END},
    {"experimento": "D", "train_start": 2021, "train_end": TRAIN_END},
]


def carregar_dados() -> tuple[pd.DataFrame, pd.DataFrame]:
    x = pd.read_csv(INPUT_X)
    y = pd.read_csv(INPUT_Y)

    if len(x) != len(y):
        raise RuntimeError(f"X e y com tamanhos diferentes: {len(x)} vs {len(y)}")

    if x.isna().sum().sum() > 0 or y.isna().sum().sum() > 0:
        raise RuntimeError("X ou y contem valores nulos.")

    return x, y


def carregar_decay() -> float:
    if not INPUT_DECAY.exists():
        return DECAY_PADRAO

    for linha in INPUT_DECAY.read_text(encoding="utf-8").splitlines():
        if linha.lower().startswith("time-decay escolhido:"):
            return float(linha.split(":", 1)[1].strip())

    return DECAY_PADRAO


def carregar_parametros_lightgbm() -> dict:
    if not INPUT_PARAMS.exists():
        raise FileNotFoundError(f"Parametros LightGBM ausentes: {INPUT_PARAMS}")

    params = json.loads(INPUT_PARAMS.read_text(encoding="utf-8"))
    params.pop("tempo_tuning_segundos", None)
    params.pop("lightgbm_version", None)
    return params


def criar_modelo(params: dict) -> LGBMRegressor:
    return LGBMRegressor(
        objective="regression",
        random_state=42,
        n_jobs=4,
        verbosity=-1,
        **params,
    )


def calcular_sample_weight(y_train: pd.DataFrame, valid_season: int, decay: float) -> np.ndarray:
    distancia = valid_season - y_train["season"]
    distancia = distancia.clip(lower=0)
    return np.power(decay, distancia).to_numpy()


def rodar_experimento(
    x: pd.DataFrame,
    y: pd.DataFrame,
    params: dict,
    decay: float,
    experimento: str,
    train_start: int,
    train_end: int,
) -> tuple[pd.DataFrame, dict]:
    train_mask = (
        (y["season"] >= train_start)
        & (y["season"] <= train_end)
    )
    test_mask = y["season"] == VALID_SEASON

    if train_mask.sum() == 0:
        raise RuntimeError(f"Nenhuma linha de treino para {train_start}-{train_end}.")
    if test_mask.sum() == 0:
        raise RuntimeError(f"Nenhuma linha de teste para {VALID_SEASON}.")

    sample_weight = calcular_sample_weight(
        y_train=y.loc[train_mask],
        valid_season=VALID_SEASON,
        decay=decay,
    )

    modelo = criar_modelo(params)
    modelo.fit(
        x.loc[train_mask],
        y.loc[train_mask, "finish_position"],
        sample_weight=sample_weight,
    )

    df_pred = y.loc[test_mask].copy()
    df_pred["pred_finish_position"] = modelo.predict(x.loc[test_mask])
    df_pred["experimento"] = experimento
    df_pred["train_start"] = train_start
    df_pred["train_end"] = train_end
    df_pred["valid_season"] = VALID_SEASON
    df_pred["decay"] = decay

    metricas = calcular_metricas(df_pred)
    metricas.update(
        {
            "experimento": experimento,
            "anos_de_treino": f"{train_start}-{train_end}",
            "train_start": train_start,
            "train_end": train_end,
            "valid_season": VALID_SEASON,
            "decay": decay,
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "corridas_treino": int(
                y.loc[train_mask, ["season", "round"]]
                .drop_duplicates()
                .shape[0]
            ),
            "corridas_teste": int(
                y.loc[test_mask, ["season", "round"]]
                .drop_duplicates()
                .shape[0]
            ),
        }
    )

    return df_pred, metricas


def gerar_relatorio(df_metricas: pd.DataFrame, params: dict, decay: float) -> None:
    tabela = df_metricas[
        [
            "experimento",
            "anos_de_treino",
            "n_train",
            "mae",
            "delta_mae_vs_a",
            "r2",
            "delta_r2_vs_a",
            "top3_accuracy",
            "delta_top3_vs_a",
            "kendall_tau",
            "rmse",
        ]
    ].copy()

    melhor_mae = tabela.sort_values("mae").iloc[0]
    melhor_r2 = tabela.sort_values("r2", ascending=False).iloc[0]
    melhor_top3 = tabela.sort_values("top3_accuracy", ascending=False).iloc[0]
    tabela_md = formatar_markdown(tabela)

    linhas = [
        "# Experimento de Janela Historica de Treino - Teste 2025",
        "",
        "## Objetivo",
        "",
        (
            "Avaliar se reduzir progressivamente o inicio da janela historica de treino "
            "provoca queda nas metricas de validacao em 2025. O teste mantem fixos o "
            "modelo, o ano de teste, as features finais, os hiperparametros e o "
            "time-decay; apenas o primeiro ano incluido no treino muda."
        ),
        "",
        "## Configuracao",
        "",
        f"- Modelo: LightGBM tunado, versao `lightgbm=={lgb.__version__}`.",
        f"- X: `{INPUT_X}`.",
        f"- y/metadados: `{INPUT_Y}`.",
        f"- Parametros: `{INPUT_PARAMS}`.",
        f"- Time-decay: `{decay}`.",
        f"- Teste fixo: `{VALID_SEASON}`.",
        "- Target: `finish_position`.",
        "- Metricas: MAE, R2, Top-3 accuracy, Kendall tau e RMSE.",
        "",
        "## Parametros do Modelo",
        "",
        "```json",
        json.dumps(params, indent=2, sort_keys=True),
        "```",
        "",
        "## Resultados",
        "",
        tabela_md,
        "",
        "Os deltas foram calculados em relacao ao experimento A. Para MAE, valor positivo indica piora; para R2 e Top-3 accuracy, valor negativo indica piora.",
        "",
        "## Leitura Objetiva",
        "",
        (
            f"- Melhor MAE: experimento {melhor_mae['experimento']} "
            f"({melhor_mae['anos_de_treino']}), MAE={melhor_mae['mae']:.6f}."
        ),
        (
            f"- Melhor R2: experimento {melhor_r2['experimento']} "
            f"({melhor_r2['anos_de_treino']}), R2={melhor_r2['r2']:.6f}."
        ),
        (
            f"- Melhor Top-3 accuracy: experimento {melhor_top3['experimento']} "
            f"({melhor_top3['anos_de_treino']}), Top-3={melhor_top3['top3_accuracy']:.6f}."
        ),
        "",
        "## Observacao Metodologica",
        "",
        (
            "Este experimento testa o efeito da quantidade de linhas usadas no ajuste "
            "do modelo. As features ja estavam previamente calculadas pelo pipeline final "
            "do projeto; portanto, o teste nao reexecuta toda a engenharia de features "
            "para cada janela. Essa escolha e coerente com a pergunta operacional: "
            "mantido o dataset final, vale treinar com 2018-2024 ou remover anos antigos?"
        ),
        "",
        "## Artefatos Gerados",
        "",
        f"- `{OUTPUT_METRICAS}`",
        f"- `{OUTPUT_PREDICOES}`",
        f"- `{OUTPUT_RELATORIO}`",
        "",
    ]

    OUTPUT_RELATORIO.write_text("\n".join(linhas), encoding="utf-8")


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


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    x, y = carregar_dados()
    params = carregar_parametros_lightgbm()
    decay = carregar_decay()

    predicoes = []
    metricas = []

    for cfg in EXPERIMENTOS:
        df_pred, met = rodar_experimento(
            x=x,
            y=y,
            params=params,
            decay=decay,
            experimento=cfg["experimento"],
            train_start=cfg["train_start"],
            train_end=cfg["train_end"],
        )
        predicoes.append(df_pred)
        metricas.append(met)

    df_predicoes = pd.concat(predicoes, ignore_index=True)
    df_metricas = pd.DataFrame(metricas)

    base_a = df_metricas.loc[df_metricas["experimento"] == "A"].iloc[0]
    df_metricas["delta_mae_vs_a"] = df_metricas["mae"] - base_a["mae"]
    df_metricas["delta_r2_vs_a"] = df_metricas["r2"] - base_a["r2"]
    df_metricas["delta_top3_vs_a"] = (
        df_metricas["top3_accuracy"] - base_a["top3_accuracy"]
    )

    ordem = [
        "experimento",
        "anos_de_treino",
        "mae",
        "delta_mae_vs_a",
        "rmse",
        "r2",
        "delta_r2_vs_a",
        "kendall_tau",
        "top3_accuracy",
        "delta_top3_vs_a",
        "train_start",
        "train_end",
        "valid_season",
        "decay",
        "n_train",
        "n_test",
        "corridas_treino",
        "corridas_teste",
    ]
    df_metricas = df_metricas[ordem]

    df_metricas.to_csv(OUTPUT_METRICAS, index=False, encoding="utf-8-sig")
    df_predicoes.to_csv(OUTPUT_PREDICOES, index=False, encoding="utf-8-sig")
    gerar_relatorio(df_metricas, params, decay)

    print("Experimento de janela de treino concluido.")
    print(df_metricas.to_string(index=False))
    print(f"\nArquivos gerados:\n- {OUTPUT_METRICAS}\n- {OUTPUT_PREDICOES}\n- {OUTPUT_RELATORIO}")


if __name__ == "__main__":
    main()
