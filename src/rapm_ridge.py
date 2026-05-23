from pathlib import Path
import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder


# 10 - Modelo auxiliar RAPM Ridge
#
# Objetivo:
# - estimar coeficientes historicos de piloto e construtor;
# - construir matriz esparsa binaria com indicadores de driver_id e constructor_id;
# - aplicar Ridge Regression com pesos temporais time-decay;
# - respeitar causalidade: para cada corrida r, treinar somente com corridas anteriores a r;
# - gerar coeficientes prontos para merge por RaceID no Feature Engineering.
#
# Observacao metodologica:
# O alvo usado e -finish_position. Assim, coeficientes maiores indicam contribuicao
# associada a melhores resultados, pois terminar em 1o lugar vira -1, 2o vira -2 etc.

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
DOCS_DIR = BASE_DIR / "docs"
MODELS_DIR = BASE_DIR / "models" / "rapm"

DEFAULT_INPUT = PROCESSED_DIR / "dataset_feature_engineering_ready_2018_2025.csv"
OUTPUT_DRIVERS = PROCESSED_DIR / "coef_pilotos_rapm_2018_2025.csv"
OUTPUT_CONSTRUCTORS = PROCESSED_DIR / "coef_construtores_rapm_2018_2025.csv"
LEGACY_OUTPUT_DRIVERS = PROCESSED_DIR / "coef_pilotos.csv"
LEGACY_OUTPUT_CONSTRUCTORS = PROCESSED_DIR / "coef_construtores.csv"
REPORT_FILE = PROCESSED_DIR / "relatorio_10_rapm_ridge.txt"
DOC_FILE = DOCS_DIR / "metodologia_rapm_ridge.md"
MANIFEST_FILE = MODELS_DIR / "manifest_rapm_ridge.json"

REQUIRED_COLUMNS = [
    "season",
    "round",
    "RaceID",
    "race_name",
    "driver_id",
    "constructor_id",
    "finish_position",
]


def repo_relative(path: Path) -> str:
    path = Path(path)
    try:
        return path.relative_to(BASE_DIR).as_posix()
    except ValueError:
        return path.as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera coeficientes historicos RAPM com Ridge Regression causal."
    )

    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="CSV de entrada com historico de corridas.",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=10.0,
        help="Regularizacao Ridge. Quanto maior, mais conservadores os coeficientes.",
    )

    parser.add_argument(
        "--decay",
        type=float,
        default=0.75,
        help="Fator de time-decay. Por padrao, aplicado por temporada.",
    )

    parser.add_argument(
        "--decay-unit",
        choices=["season", "race"],
        default="season",
        help="Unidade do time-decay: season segue o plano/RAPM; race permite decaimento por corrida.",
    )

    parser.add_argument(
        "--min-races-train",
        type=int,
        default=1,
        help="Minimo de corridas historicas antes de estimar coeficientes. Antes disso, usa cold start 0.0.",
    )

    parser.add_argument(
        "--loess",
        action="store_true",
        help="Aplica suavizacao LOESS opcional aos coeficientes por entidade.",
    )

    parser.add_argument(
        "--loess-frac",
        type=float,
        default=0.30,
        help="Parametro frac do LOESS. Usado somente com --loess.",
    )

    return parser.parse_args()


def load_and_validate(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo de entrada nao encontrado: {input_path}")

    df = pd.read_csv(input_path)

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Colunas obrigatorias ausentes: {missing}")

    df = df.copy()
    df = df[REQUIRED_COLUMNS].dropna(subset=REQUIRED_COLUMNS)

    df["season"] = df["season"].astype(int)
    df["round"] = df["round"].astype(int)
    df["finish_position"] = pd.to_numeric(df["finish_position"], errors="coerce")

    df = df.dropna(subset=["finish_position"])
    df = df[df["finish_position"] > 0]

    # Garante uma linha por piloto em cada corrida sem perder a chave de merge.
    df = df.drop_duplicates(subset=["RaceID"])

    df = df.sort_values(
        ["season", "round", "finish_position", "driver_id"]
    ).reset_index(drop=True)

    race_order = (
        df[["season", "round", "race_name"]]
        .drop_duplicates()
        .sort_values(["season", "round"])
        .reset_index(drop=True)
    )

    race_order["race_order"] = np.arange(1, len(race_order) + 1)

    df = df.merge(
        race_order,
        on=["season", "round", "race_name"],
        how="left",
    )

    return df


def build_sparse_matrix(df_train: pd.DataFrame):
    driver_encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=True,
        dtype=np.float64,
    )

    constructor_encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=True,
        dtype=np.float64,
    )

    x_driver = driver_encoder.fit_transform(df_train[["driver_id"]])
    x_constructor = constructor_encoder.fit_transform(df_train[["constructor_id"]])

    x = hstack([x_driver, x_constructor], format="csr")

    driver_names = driver_encoder.get_feature_names_out(["driver_id"])
    constructor_names = constructor_encoder.get_feature_names_out(["constructor_id"])

    feature_names = np.concatenate([driver_names, constructor_names])

    return x, feature_names


def time_decay_weights(
    df_train: pd.DataFrame,
    current_race_order: int,
    current_season: int,
    decay: float,
    decay_unit: str,
) -> np.ndarray:
    if decay_unit == "season":
        distance = current_season - df_train["season"].to_numpy(dtype=float)
        distance = np.maximum(distance, 0.0)
    else:
        distance = current_race_order - df_train["race_order"].to_numpy(dtype=float)
        distance = np.maximum(distance, 1.0)

    weights = np.power(decay, distance)

    return weights


def fit_ridge(
    df_train: pd.DataFrame,
    alpha: float,
    decay: float,
    current_race_order: int,
    current_season: int,
    decay_unit: str,
):
    x, feature_names = build_sparse_matrix(df_train)

    # Target invertido:
    # finish_position menor é melhor.
    # Por isso usamos -finish_position para coeficiente maior representar melhor desempenho.
    y = -df_train["finish_position"].to_numpy(dtype=float)

    weights = time_decay_weights(
        df_train,
        current_race_order=current_race_order,
        current_season=current_season,
        decay=decay,
        decay_unit=decay_unit,
    )

    model = Ridge(
        alpha=alpha,
        fit_intercept=True,
        solver="auto",
        random_state=42,
    )

    model.fit(x, y, sample_weight=weights)

    coef_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coef_rapm": model.coef_,
        }
    )

    return model, coef_df


def split_coefficients(coef_df: pd.DataFrame):
    drivers = coef_df[coef_df["feature"].str.startswith("driver_id_")].copy()
    constructors = coef_df[coef_df["feature"].str.startswith("constructor_id_")].copy()

    drivers["driver_id"] = drivers["feature"].str.replace(
        "driver_id_",
        "",
        regex=False,
    )

    constructors["constructor_id"] = constructors["feature"].str.replace(
        "constructor_id_",
        "",
        regex=False,
    )

    drivers = drivers[["driver_id", "coef_rapm"]]
    constructors = constructors[["constructor_id", "coef_rapm"]]

    return drivers, constructors


def smooth_loess(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    frac: float,
) -> pd.DataFrame:
    df = df.copy()
    smooth_col = f"{value_col}_loess"
    df[smooth_col] = np.nan

    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
    except Exception:
        df[smooth_col] = df.groupby(group_col)[value_col].transform(
            lambda s: s.rolling(window=3, min_periods=1).mean()
        )
        df["loess_status"] = "statsmodels_indisponivel_rolling_media_3"
        return df

    for _, idx in df.groupby(group_col).groups.items():
        idx = list(idx)
        serie = df.loc[idx].sort_values("race_order")

        if len(serie) < 4:
            df.loc[serie.index, smooth_col] = serie[value_col].to_numpy()
            continue

        smoothed = lowess(
            endog=serie[value_col].to_numpy(dtype=float),
            exog=serie["race_order"].to_numpy(dtype=float),
            frac=frac,
            return_sorted=False,
        )

        df.loc[serie.index, smooth_col] = smoothed

    df["loess_status"] = "loess_aplicado"

    return df


def generate_rapm(
    df: pd.DataFrame,
    alpha: float,
    decay: float,
    decay_unit: str,
    min_races_train: int,
    apply_loess: bool,
    loess_frac: float,
):
    race_table = (
        df[["season", "round", "race_name", "race_order"]]
        .drop_duplicates()
        .sort_values("race_order")
        .reset_index(drop=True)
    )

    driver_rows = []
    constructor_rows = []
    skipped = []

    for race in race_table.itertuples(index=False):
        current_order = int(race.race_order)
        current_season = int(race.season)
        current_rows = (
            df[df["race_order"] == current_order]
            .sort_values(["finish_position", "driver_id"])
            .copy()
        )

        # Causalidade:
        # para a corrida atual, treina somente com corridas anteriores.
        train = df[df["race_order"] < current_order].copy()

        n_train_races = int(
            train[["season", "round"]].drop_duplicates().shape[0]
        )

        if n_train_races < min_races_train:
            skipped.append(
                {
                    "season": current_season,
                    "round": int(race.round),
                    "race_name": race.race_name,
                    "race_order": current_order,
                    "motivo": f"cold_start_menos_de_{min_races_train}_corridas_anteriores",
                }
            )
            driver_map = {}
            constructor_map = {}
            coefficient_status = "cold_start_sem_historico_suficiente"
        else:
            _, coef_df = fit_ridge(
                train,
                alpha=alpha,
                decay=decay,
                current_race_order=current_order,
                current_season=current_season,
                decay_unit=decay_unit,
            )

            drivers, constructors = split_coefficients(coef_df)
            driver_map = drivers.set_index("driver_id")["coef_rapm"].to_dict()
            constructor_map = constructors.set_index("constructor_id")["coef_rapm"].to_dict()
            coefficient_status = "estimado_historico_anterior"

        drivers_current = current_rows[
            ["season", "round", "RaceID", "race_name", "driver_id", "constructor_id", "race_order"]
        ].copy()
        drivers_current["driver_coef_rapm"] = (
            drivers_current["driver_id"].map(driver_map).fillna(0.0)
        )
        drivers_current["rapm_cold_start_flag"] = (
            ~drivers_current["driver_id"].isin(driver_map.keys())
        ).astype(int)

        constructors_current = current_rows[
            ["season", "round", "RaceID", "race_name", "driver_id", "constructor_id", "race_order"]
        ].copy()
        constructors_current["constructor_coef_rapm"] = (
            constructors_current["constructor_id"].map(constructor_map).fillna(0.0)
        )
        constructors_current["rapm_cold_start_flag"] = (
            ~constructors_current["constructor_id"].isin(constructor_map.keys())
        ).astype(int)

        for fixed_df in [drivers_current, constructors_current]:
            fixed_df["target_definition"] = "-finish_position"
            fixed_df["alpha"] = alpha
            fixed_df["decay"] = decay
            fixed_df["decay_unit"] = decay_unit
            fixed_df["coefficient_status"] = coefficient_status
            fixed_df["n_train_rows"] = int(len(train))
            fixed_df["n_train_races"] = n_train_races

        driver_rows.append(drivers_current)
        constructor_rows.append(constructors_current)

    coef_pilotos = (
        pd.concat(driver_rows, ignore_index=True)
        if driver_rows
        else pd.DataFrame()
    )

    coef_construtores = (
        pd.concat(constructor_rows, ignore_index=True)
        if constructor_rows
        else pd.DataFrame()
    )

    if apply_loess and not coef_pilotos.empty:
        coef_pilotos = smooth_loess(
            coef_pilotos,
            "driver_id",
            "driver_coef_rapm",
            loess_frac,
        )

    if apply_loess and not coef_construtores.empty:
        coef_construtores = smooth_loess(
            coef_construtores,
            "constructor_id",
            "constructor_coef_rapm",
            loess_frac,
        )

    skipped_df = pd.DataFrame(skipped)

    return coef_pilotos, coef_construtores, skipped_df, race_table


def write_report(
    df: pd.DataFrame,
    coef_pilotos: pd.DataFrame,
    coef_construtores: pd.DataFrame,
    skipped_df: pd.DataFrame,
    race_table: pd.DataFrame,
    args: argparse.Namespace,
):
    linhas = []

    linhas.append("Relatorio 10 - RAPM Ridge")
    linhas.append("=" * 32)
    linhas.append(f"Gerado em: {datetime.now().isoformat(timespec='seconds')}")
    linhas.append(f"Entrada: {repo_relative(Path(args.input))}")
    linhas.append(f"Temporadas na base: {int(df['season'].min())}-{int(df['season'].max())}")
    linhas.append(f"Total de linhas historicas: {len(df)}")
    linhas.append(f"Total de corridas: {race_table.shape[0]}")
    linhas.append(f"Alpha Ridge: {args.alpha}")
    linhas.append(f"Time-decay: {args.decay}")
    linhas.append(f"Unidade do time-decay: {args.decay_unit}")
    linhas.append(f"Minimo de corridas para treino: {args.min_races_train}")
    linhas.append(f"LOESS solicitado: {bool(args.loess)}")
    linhas.append("")
    linhas.append("Saidas geradas:")
    linhas.append(f"- {repo_relative(OUTPUT_DRIVERS)}: {len(coef_pilotos)} linhas")
    linhas.append(f"- {repo_relative(OUTPUT_CONSTRUCTORS)}: {len(coef_construtores)} linhas")
    linhas.append("")
    linhas.append("Logica causal:")
    linhas.append("Para cada corrida r, o modelo foi treinado apenas com corridas anteriores a r.")
    linhas.append("A corrida r recebe o snapshot de coeficientes conhecido antes dela acontecer.")
    linhas.append("")
    linhas.append("Interpretacao:")
    linhas.append("O target e -finish_position. Portanto, coeficientes maiores indicam melhor contribuicao historica estimada.")
    linhas.append("")
    linhas.append("Corridas com cold start por falta de historico inicial:")
    linhas.append(str(len(skipped_df)))

    REPORT_FILE.write_text("\n".join(linhas), encoding="utf-8")


def write_doc(args: argparse.Namespace):
    text = f"""# Metodologia - RAPM Ridge

## Objetivo

Esta etapa cria coeficientes auxiliares de desempenho para pilotos e construtores usando uma abordagem inspirada em RAPM, com Ridge Regression e matriz esparsa binaria.

## Entrada

Arquivo padrao:

{repo_relative(DEFAULT_INPUT)}

Colunas obrigatorias:

- season
- round
- RaceID
- race_name
- driver_id
- constructor_id
- finish_position

## Matriz do modelo

Para cada linha piloto-corrida, o script cria indicadores binarios para:

- piloto
- construtor

A matriz e mantida como esparsa para evitar aumento desnecessario de memoria.

## Target

O alvo usado no modelo e:

target = -finish_position

Com isso, coeficientes maiores representam associacao com melhores resultados historicos.

## Causalidade

A geracao e feita corrida a corrida.

Para cada corrida r, o modelo e treinado somente com corridas anteriores a r.

Assim, os coeficientes podem ser usados como feature historica sem vazamento de informacao futura.

Para corridas ou entidades sem historico suficiente, o coeficiente recebe cold start 0.0.

## Time-decay

O peso de cada observacao historica e calculado por distancia temporal:

peso = decay ^ distancia

Unidade padrao:

decay_unit = {args.decay_unit}

Valor de decay:

decay = {args.decay}

Corridas mais recentes recebem maior peso.

## Regularizacao Ridge

A regressao usa Ridge para reduzir instabilidade dos coeficientes.

Valor padrao:

alpha = {args.alpha}

## Saidas

data/processed/coef_pilotos_rapm_2018_2025.csv
data/processed/coef_construtores_rapm_2018_2025.csv
data/processed/relatorio_10_rapm_ridge.txt
models/rapm/manifest_rapm_ridge.json

Tambem sao gravadas copias de compatibilidade em:

data/processed/coef_pilotos.csv
data/processed/coef_construtores.csv

## LOESS opcional

O script permite suavizar os coeficientes com LOESS usando:

python src/rapm_ridge.py --loess
"""

    DOC_FILE.write_text(text, encoding="utf-8")


def write_manifest(
    df: pd.DataFrame,
    coef_pilotos: pd.DataFrame,
    coef_construtores: pd.DataFrame,
    skipped_df: pd.DataFrame,
    race_table: pd.DataFrame,
    args: argparse.Namespace,
):
    manifest = {
        "step": "10_rapm_ridge",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input": repo_relative(Path(args.input)),
        "outputs": {
            "drivers": repo_relative(OUTPUT_DRIVERS),
            "constructors": repo_relative(OUTPUT_CONSTRUCTORS),
            "drivers_legacy_copy": repo_relative(LEGACY_OUTPUT_DRIVERS),
            "constructors_legacy_copy": repo_relative(LEGACY_OUTPUT_CONSTRUCTORS),
            "report": repo_relative(REPORT_FILE),
            "documentation": repo_relative(DOC_FILE),
        },
        "target": "-finish_position",
        "alpha": args.alpha,
        "decay": args.decay,
        "decay_unit": args.decay_unit,
        "min_races_train": args.min_races_train,
        "loess": bool(args.loess),
        "loess_frac": args.loess_frac,
        "n_rows_input": int(len(df)),
        "season_min": int(df["season"].min()),
        "season_max": int(df["season"].max()),
        "n_races": int(race_table.shape[0]),
        "n_driver_coef_rows": int(len(coef_pilotos)),
        "n_constructor_coef_rows": int(len(coef_construtores)),
        "n_cold_start_initial_races": int(len(skipped_df)),
        "anti_leakage_rule": "for each race r, train only on races before r",
        "merge_contract": {
            "drivers": ["season", "round", "RaceID", "driver_id", "driver_coef_rapm"],
            "constructors": ["season", "round", "RaceID", "constructor_id", "constructor_coef_rapm"],
        },
    }

    MANIFEST_FILE.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main():
    args = parse_args()

    input_path = Path(args.input)

    if not input_path.is_absolute():
        input_path = BASE_DIR / input_path

    args.input = str(input_path)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_validate(input_path)

    coef_pilotos, coef_construtores, skipped_df, race_table = generate_rapm(
        df=df,
        alpha=args.alpha,
        decay=args.decay,
        decay_unit=args.decay_unit,
        min_races_train=args.min_races_train,
        apply_loess=args.loess,
        loess_frac=args.loess_frac,
    )

    if coef_pilotos.empty or coef_construtores.empty:
        raise RuntimeError(
            "Nao foi possivel gerar coeficientes. Verifique o tamanho da base e min_races_train."
        )

    coef_pilotos = coef_pilotos.sort_values(
        ["season", "round", "RaceID", "driver_id"]
    ).reset_index(drop=True)

    coef_construtores = coef_construtores.sort_values(
        ["season", "round", "RaceID", "constructor_id"]
    ).reset_index(drop=True)

    coef_pilotos.to_csv(OUTPUT_DRIVERS, index=False)
    coef_construtores.to_csv(OUTPUT_CONSTRUCTORS, index=False)
    coef_pilotos.to_csv(LEGACY_OUTPUT_DRIVERS, index=False)
    coef_construtores.to_csv(LEGACY_OUTPUT_CONSTRUCTORS, index=False)

    write_report(
        df,
        coef_pilotos,
        coef_construtores,
        skipped_df,
        race_table,
        args,
    )

    write_doc(args)

    write_manifest(
        df,
        coef_pilotos,
        coef_construtores,
        skipped_df,
        race_table,
        args,
    )

    print("RAPM Ridge concluido com sucesso.")
    print(f"Entrada: {input_path}")
    print(f"Coeficientes pilotos: {OUTPUT_DRIVERS}")
    print(f"Coeficientes construtores: {OUTPUT_CONSTRUCTORS}")
    print(f"Relatorio: {REPORT_FILE}")
    print(f"Documentacao: {DOC_FILE}")
    print(f"Manifest: {MANIFEST_FILE}")


if __name__ == "__main__":
    main()
