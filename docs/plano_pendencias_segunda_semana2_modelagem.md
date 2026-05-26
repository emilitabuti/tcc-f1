# Pendencias da Segunda - Semana 2

## Objetivo

Este documento detalha o que ainda precisa ser feito para concluir a segunda-feira da Semana 2 do cronograma de modelagem.

Escopo considerado:

- Implementar validacao walk-forward.
- Integrar o fold de validacao 2025.
- Criar metricas customizadas.
- Otimizar o fator de time-decay.
- Aplicar o time-decay no XGBoost.

Fora do escopo por enquanto:

- Random Forest.
- Tuning Optuna.
- Ridge baseline.
- RFE final.

## Situacao Atual

Ja existem bases prontas para modelagem:

- `data/processed/dataset_modelagem_X_2018_2025.csv`
- `data/processed/dataset_modelagem_y_2018_2025.csv`
- `data/processed/dataset_modelagem_2018_2025.csv`
- `data/processed/openf1_2025_clean.csv`
- `data/processed/validacao_2025_clean.csv`

As bases `X` e `y` estao alinhadas, sem nulos, com temporadas de 2018 a 2025.

Tambem ja existe uma validacao temporal parcial no script:

- `src/rfe_xgboost_features.py`

Esse script treina em 2018-2024 e valida em 2025, mas isso ainda nao substitui o walk-forward completo da segunda-feira.

## O Que Falta Fazer

### 1. Criar `src/metricas.py`

O cronograma pede um arquivo de metricas customizadas com:

- MAE
- RMSE
- R2
- Kendall tau medio por corrida
- Acuracia top-3
- Wrapper geral calculando tudo de uma vez

Esse arquivo deve ser independente dos modelos, para ser usado tanto no XGBoost agora quanto nos outros modelos depois.

Codigo sugerido:

```python
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calcular_mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def calcular_rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def calcular_r2(y_true, y_pred) -> float:
    return float(r2_score(y_true, y_pred))


def kendall_tau_por_corrida(df_pred: pd.DataFrame) -> float:
    valores = []

    for _, grupo in df_pred.groupby(["season", "round"]):
        if grupo["finish_position"].nunique() < 2:
            continue

        tau, _ = kendalltau(
            grupo["finish_position"],
            grupo["pred_finish_position"],
        )

        if not np.isnan(tau):
            valores.append(tau)

    if not valores:
        return float("nan")

    return float(np.mean(valores))


def acuracia_top3(df_pred: pd.DataFrame) -> float:
    acertos = []

    for _, grupo in df_pred.groupby(["season", "round"]):
        real_top3 = set(
            grupo.sort_values("finish_position")
            .head(3)["driver_id"]
            .tolist()
        )

        pred_top3 = set(
            grupo.sort_values("pred_finish_position")
            .head(3)["driver_id"]
            .tolist()
        )

        acertos.append(int(real_top3 == pred_top3))

    if not acertos:
        return float("nan")

    return float(np.mean(acertos))


def calcular_metricas(df_pred: pd.DataFrame) -> dict:
    y_true = df_pred["finish_position"]
    y_pred = df_pred["pred_finish_position"]

    return {
        "mae": calcular_mae(y_true, y_pred),
        "rmse": calcular_rmse(y_true, y_pred),
        "r2": calcular_r2(y_true, y_pred),
        "kendall_tau": kendall_tau_por_corrida(df_pred),
        "top3_accuracy": acuracia_top3(df_pred),
    }
```

### 2. Criar `src/walk_forward.py`

O cronograma pede estes folds:

- Treino 2014-2022 -> validacao 2023
- Treino 2014-2023 -> validacao 2024
- Treino 2014-2024 -> validacao 2025

No projeto atual, o recorte oficial dos dados esta em 2018-2025. Entao, na implementacao real, os folds devem ficar assim:

- Treino 2018-2022 -> validacao 2023
- Treino 2018-2023 -> validacao 2024
- Treino 2018-2024 -> validacao 2025

Esse ajuste precisa ser documentado no relatorio, porque o cronograma menciona 2014, mas a base final esta filtrada para 2018 em diante.

Codigo sugerido:

```python
from __future__ import annotations

from pathlib import Path

import pandas as pd
from xgboost import XGBRegressor

from metricas import calcular_metricas


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports" / "modelagem"

INPUT_X = PROCESSED_DIR / "dataset_modelagem_X_2018_2025.csv"
INPUT_Y = PROCESSED_DIR / "dataset_modelagem_y_2018_2025.csv"

OUTPUT_PREDICOES = REPORTS_DIR / "predicoes_walk_forward_xgboost.csv"
OUTPUT_METRICAS = REPORTS_DIR / "metricas_walk_forward_xgboost.csv"


FOLDS = [
    {"train_until": 2022, "valid_season": 2023},
    {"train_until": 2023, "valid_season": 2024},
    {"train_until": 2024, "valid_season": 2025},
]


def carregar_dados():
    x = pd.read_csv(INPUT_X)
    y = pd.read_csv(INPUT_Y)

    if len(x) != len(y):
        raise RuntimeError(f"X e y com tamanhos diferentes: {len(x)} vs {len(y)}")

    return x, y


def criar_modelo_xgboost() -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=350,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        reg_alpha=0.0,
        random_state=42,
        n_jobs=4,
    )


def rodar_fold(x, y, train_until: int, valid_season: int):
    train_mask = y["season"] <= train_until
    valid_mask = y["season"] == valid_season

    if valid_mask.sum() == 0:
        raise RuntimeError(f"Nenhuma linha encontrada para validacao {valid_season}.")

    modelo = criar_modelo_xgboost()
    modelo.fit(
        x.loc[train_mask],
        y.loc[train_mask, "finish_position"],
    )

    pred = modelo.predict(x.loc[valid_mask])

    df_pred = y.loc[valid_mask].copy()
    df_pred["pred_finish_position"] = pred
    df_pred["train_until"] = train_until
    df_pred["valid_season"] = valid_season

    metricas = calcular_metricas(df_pred)
    metricas["train_until"] = train_until
    metricas["valid_season"] = valid_season
    metricas["n_train"] = int(train_mask.sum())
    metricas["n_valid"] = int(valid_mask.sum())

    return df_pred, metricas


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    x, y = carregar_dados()

    predicoes = []
    metricas = []

    for fold in FOLDS:
        df_pred, fold_metricas = rodar_fold(
            x=x,
            y=y,
            train_until=fold["train_until"],
            valid_season=fold["valid_season"],
        )

        predicoes.append(df_pred)
        metricas.append(fold_metricas)

    df_predicoes = pd.concat(predicoes, ignore_index=True)
    df_metricas = pd.DataFrame(metricas)

    df_predicoes.to_csv(OUTPUT_PREDICOES, index=False, encoding="utf-8-sig")
    df_metricas.to_csv(OUTPUT_METRICAS, index=False, encoding="utf-8-sig")

    print("Walk-forward XGBoost concluido.")
    print(df_metricas.to_string(index=False))


if __name__ == "__main__":
    main()
```

### 3. Criar `src/otimizacao_time_decay.py`

O cronograma pede testar os fatores:

- 0.50
- 0.65
- 0.75
- 0.85
- 0.95

A ideia e calcular pesos temporais para cada linha de treino e passar esses pesos ao `sample_weight` do XGBoost.

Como a validacao pedida para escolher o fator e 2023-2024, o script deve:

1. Rodar fold 2023 para cada fator.
2. Rodar fold 2024 para cada fator.
3. Calcular o MAE medio.
4. Escolher o fator com menor MAE medio.
5. Salvar o resultado em CSV.

Codigo sugerido:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from metricas import calcular_metricas


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports" / "modelagem"

INPUT_X = PROCESSED_DIR / "dataset_modelagem_X_2018_2025.csv"
INPUT_Y = PROCESSED_DIR / "dataset_modelagem_y_2018_2025.csv"

OUTPUT_RESULTADOS = REPORTS_DIR / "otimizacao_time_decay_xgboost.csv"
OUTPUT_ESCOLHIDO = REPORTS_DIR / "time_decay_escolhido_xgboost.txt"

DECAYS = [0.50, 0.65, 0.75, 0.85, 0.95]
FOLDS_OTIMIZACAO = [
    {"train_until": 2022, "valid_season": 2023},
    {"train_until": 2023, "valid_season": 2024},
]


def carregar_dados():
    x = pd.read_csv(INPUT_X)
    y = pd.read_csv(INPUT_Y)

    if len(x) != len(y):
        raise RuntimeError(f"X e y com tamanhos diferentes: {len(x)} vs {len(y)}")

    return x, y


def criar_modelo_xgboost() -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=350,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        reg_alpha=0.0,
        random_state=42,
        n_jobs=4,
    )


def calcular_sample_weight(y_train: pd.DataFrame, valid_season: int, decay: float):
    distancia = valid_season - y_train["season"]
    distancia = distancia.clip(lower=0)
    pesos = np.power(decay, distancia)
    return pesos.to_numpy()


def avaliar_decay(x, y, decay: float, train_until: int, valid_season: int):
    train_mask = y["season"] <= train_until
    valid_mask = y["season"] == valid_season

    x_train = x.loc[train_mask]
    y_train = y.loc[train_mask]
    x_valid = x.loc[valid_mask]
    y_valid = y.loc[valid_mask].copy()

    sample_weight = calcular_sample_weight(
        y_train=y_train,
        valid_season=valid_season,
        decay=decay,
    )

    modelo = criar_modelo_xgboost()
    modelo.fit(
        x_train,
        y_train["finish_position"],
        sample_weight=sample_weight,
    )

    y_valid["pred_finish_position"] = modelo.predict(x_valid)
    metricas = calcular_metricas(y_valid)

    return {
        "decay": decay,
        "train_until": train_until,
        "valid_season": valid_season,
        "n_train": int(train_mask.sum()),
        "n_valid": int(valid_mask.sum()),
        **metricas,
    }


def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    x, y = carregar_dados()

    resultados = []

    for decay in DECAYS:
        for fold in FOLDS_OTIMIZACAO:
            resultado = avaliar_decay(
                x=x,
                y=y,
                decay=decay,
                train_until=fold["train_until"],
                valid_season=fold["valid_season"],
            )
            resultados.append(resultado)

    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv(OUTPUT_RESULTADOS, index=False, encoding="utf-8-sig")

    resumo = (
        df_resultados
        .groupby("decay", as_index=False)
        .agg(
            mae_medio=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_medio=("rmse", "mean"),
            kendall_tau_medio=("kendall_tau", "mean"),
            top3_accuracy_medio=("top3_accuracy", "mean"),
        )
        .sort_values(["mae_medio", "decay"])
    )

    melhor = resumo.iloc[0]

    texto = (
        f"Time-decay escolhido: {melhor['decay']}\n"
        f"MAE medio 2023-2024: {melhor['mae_medio']:.6f}\n"
        f"Desvio padrao do MAE: {melhor['mae_std']:.6f}\n"
    )

    OUTPUT_ESCOLHIDO.write_text(texto, encoding="utf-8")

    print("Otimizacao de time-decay concluida.")
    print(resumo.to_string(index=False))
    print()
    print(texto)


if __name__ == "__main__":
    main()
```

### 4. Atualizar o walk-forward para usar o decay escolhido

Depois de escolher o melhor decay, o `walk_forward.py` deve ser atualizado para usar `sample_weight`.

A alteracao principal e incluir uma funcao de pesos:

```python
def calcular_sample_weight(y_train: pd.DataFrame, valid_season: int, decay: float):
    distancia = valid_season - y_train["season"]
    distancia = distancia.clip(lower=0)
    return np.power(decay, distancia).to_numpy()
```

E mudar o `fit` para:

```python
sample_weight = calcular_sample_weight(
    y_train=y.loc[train_mask],
    valid_season=valid_season,
    decay=decay_escolhido,
)

modelo.fit(
    x.loc[train_mask],
    y.loc[train_mask, "finish_position"],
    sample_weight=sample_weight,
)
```

O fold final 2025 deve usar o decay escolhido com base nos folds 2023 e 2024.

### 5. Gerar relatorio da segunda-feira

Ao final, criar um relatorio simples em:

- `reports/modelagem/relatorio_segunda_semana2_xgboost.txt`

Conteudo minimo:

- Bases usadas.
- Recorte temporal real: 2018-2025.
- Justificativa do ajuste de 2014 para 2018.
- Folds usados.
- Metricas por fold.
- Decay testado.
- Decay escolhido.
- Observacao de que Random Forest ficou fora por decisao temporaria.

Modelo de texto:

```text
Relatorio - Segunda Semana 2 - Modelagem XGBoost
================================================

Bases utilizadas:
- data/processed/dataset_modelagem_X_2018_2025.csv
- data/processed/dataset_modelagem_y_2018_2025.csv

Recorte temporal:
O cronograma menciona treino a partir de 2014, mas a base final de modelagem
esta filtrada para 2018-2025. Portanto, os folds foram adaptados para:
- treino 2018-2022 -> validacao 2023
- treino 2018-2023 -> validacao 2024
- treino 2018-2024 -> validacao 2025

Metricas calculadas:
- MAE
- RMSE
- R2
- Kendall tau medio por corrida
- Acuracia top-3

Time-decay:
Foram testados os fatores 0.50, 0.65, 0.75, 0.85 e 0.95.
O fator escolhido foi aquele com menor MAE medio nos folds 2023 e 2024.

Random Forest:
Nao executado nesta etapa por decisao temporaria.
```

## Ordem Recomendada De Execucao

1. Criar `src/metricas.py`.
2. Criar `src/walk_forward.py` sem time-decay para validar que tudo roda.
3. Criar `src/otimizacao_time_decay.py`.
4. Rodar a otimizacao do decay.
5. Atualizar `src/walk_forward.py` para usar o decay escolhido.
6. Rodar o walk-forward final com XGBoost.
7. Gerar o relatorio da segunda-feira.

## Criterio Para Considerar A Segunda Concluida

A segunda-feira pode ser considerada concluida quando existirem:

- `src/metricas.py`
- `src/walk_forward.py`
- `src/otimizacao_time_decay.py`
- `reports/modelagem/otimizacao_time_decay_xgboost.csv`
- `reports/modelagem/time_decay_escolhido_xgboost.txt`
- `reports/modelagem/metricas_walk_forward_xgboost.csv`
- `reports/modelagem/predicoes_walk_forward_xgboost.csv`
- `reports/modelagem/relatorio_segunda_semana2_xgboost.txt`

E quando o relatorio mostrar metricas para:

- validacao 2023
- validacao 2024
- validacao 2025

## Observacao Importante

O arquivo `src/rfe_xgboost_features.py` ja possui uma validacao temporal em 2025 e pode servir como referencia, mas ele nao substitui a segunda-feira porque:

- nao roda os folds 2023 e 2024;
- nao calcula todas as metricas customizadas;
- nao otimiza time-decay;
- nao salva uma tabela completa de metricas walk-forward.

Portanto, ele deve ser tratado como evidencia de que o XGBoost consegue rodar, nao como entrega final da segunda.
