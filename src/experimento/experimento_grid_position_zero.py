from pathlib import Path
import json
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
REPORTS_DIR = ROOT_DIR / "reports" / "modelagem"

DATASET_FULL = PROCESSED_DIR / "dataset_features_final_2018_2025.csv"
PARAMS_FILE = REPORTS_DIR / "optuna_lightgbm_best_params.json"

SAIDA = REPORTS_DIR / "experimento_grid_position_zero_metricas.csv"


def top3_accuracy_por_corrida(df_resultado):
    acertos = []
    for _, grupo in df_resultado.groupby(["season", "round"]):
        reais_top3 = set(
            grupo.sort_values("finish_position")
            .head(3)["RaceID"]
        )

        pred_top3 = set(
            grupo.sort_values("pred_finish_position")
            .head(3)["RaceID"]
        )

        acertos.append(len(reais_top3 & pred_top3) / 3)

    return float(np.mean(acertos))


def preparar_xy(df, features):
    X = df[features].copy()
    y = df["finish_position"].copy()
    return X, y


def treinar_avaliar(nome, df, features, params):
    treino = df[df["season"] <= 2024].copy()
    teste = df[df["season"] == 2025].copy()

    X_train, y_train = preparar_xy(treino, features)
    X_test, y_test = preparar_xy(teste, features)

    modelo = LGBMRegressor(**params)
    modelo.fit(X_train, y_train)

    pred = modelo.predict(X_test)

    resultado = teste[["RaceID", "season", "round", "finish_position"]].copy()
    resultado["pred_finish_position"] = pred

    mae = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    r2 = r2_score(y_test, pred)
    top3 = top3_accuracy_por_corrida(resultado)

    return {
        "cenario": nome,
        "linhas_treino": len(treino),
        "linhas_teste": len(teste),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Top3_accuracy": top3,
    }


def main():
    df = pd.read_csv(DATASET_FULL)

    with open(PARAMS_FILE, encoding="utf-8") as f:
        params = json.load(f)

    # Remover parâmetros que podem dar conflito dependendo de como foram salvos
    params.pop("objective", None)
    params.pop("metric", None)

    # Features finais: remove colunas que não devem entrar como X
    colunas_excluir = {
        "RaceID",
        "season",
        "round",
        "finish_position",
        "position_order",
        "driver_id",
        "constructor_id",
        "race_name",
        "status",
    }

    features_base = [
        col for col in df.columns
        if col not in colunas_excluir
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    # Cenário A: atual, sem grid_position e sem flag
    features_atual = [
        col for col in features_base
        if col not in ["grid_position", "grid_position_zero_flag"]
    ]

    # Cenário B: remove registros que tinham grid_position_zero_flag = 1
    df_sem_grid_zero = df[df["grid_position_zero_flag"] != 1].copy()

    # Cenário C: inclui a flag, mas não inclui grid_position
    features_com_flag = [
        col for col in features_base
        if col != "grid_position"
    ]

    resultados = []

    resultados.append(
        treinar_avaliar(
            "A_atual_sem_grid_position_sem_flag",
            df,
            features_atual,
            params,
        )
    )

    resultados.append(
        treinar_avaliar(
            "B_remover_registros_grid_zero",
            df_sem_grid_zero,
            features_atual,
            params,
        )
    )

    resultados.append(
        treinar_avaliar(
            "C_incluir_grid_position_zero_flag",
            df,
            features_com_flag,
            params,
        )
    )

    resultado_df = pd.DataFrame(resultados)
    resultado_df.to_csv(SAIDA, index=False)

    print("\n=== EXPERIMENTO grid_position = 0 ===\n")
    print(resultado_df.to_string(index=False))
    print(f"\nArquivo salvo em: {SAIDA}")


if __name__ == "__main__":
    main()