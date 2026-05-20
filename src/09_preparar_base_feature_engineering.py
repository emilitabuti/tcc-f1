from pathlib import Path
import json

import pandas as pd


# 09 - Preparacao final da base para Feature Engineering
#
# Objetivo:
# - congelar a base oficial pre-FE;
# - remover colunas que nao devem circular como features do modelo;
# - manter finish_position como target/historico para calculo causal de features;
# - gerar manifest explicito anti-leakage;
# - gerar tabela de outliers em revisao para decisao antes da modelagem.
BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "data" / "processed"
DOCS_DIR = BASE_DIR / "docs"

INPUT_2018_2024 = PROCESSED_DIR / "dataset_pre_features_2018_2024.csv"
INPUT_2018_2025 = PROCESSED_DIR / "dataset_pre_features_2018_2025.csv"

OUTPUT_FE_2018_2024 = PROCESSED_DIR / "dataset_feature_engineering_ready_2018_2024.csv"
OUTPUT_FE_2018_2025 = PROCESSED_DIR / "dataset_feature_engineering_ready_2018_2025.csv"

TARGET_2018_2024 = PROCESSED_DIR / "target_finish_position_2018_2024.csv"
TARGET_2018_2025 = PROCESSED_DIR / "target_finish_position_2018_2025.csv"

OUTLIERS_REVISAO = PROCESSED_DIR / "outliers_revisao_2018_2025.csv"
MANIFEST_FILE = PROCESSED_DIR / "manifest_feature_engineering.json"
REPORT_FILE = PROCESSED_DIR / "relatorio_09_preparacao_feature_engineering.txt"
DOC_FILE = DOCS_DIR / "metodologia_preparacao_feature_engineering.md"

TARGET = "finish_position"

# Colunas que nunca podem entrar em X. finish_position fica na base FE-ready
# porque e target e tambem insumo historico para features causais como recent_form.
COLUNAS_PROIBIDAS_MODELO = [
    "finish_position",
    "points",
    "race_points",
    "fastest_lap_race",
    "previous_position",
]

# Removidas da base FE-ready para reduzir chance de uso acidental.
# finish_position e preservada como target/historico; points nao e necessario
# para as features planejadas e e altamente colinear com o resultado.
COLUNAS_REMOVER_DA_BASE_FE = [
    "points",
    "race_points",
    "fastest_lap_race",
    "previous_position",
]

COLUNAS_CHAVE = [
    "season",
    "round",
    "RaceID",
    "driver_id",
    "constructor_id",
    "race_name",
    "circuit_id",
]

COLUNAS_OBRIGATORIAS = COLUNAS_CHAVE + [
    "grid_position",
    TARGET,
    "laps",
    "compound_ordinal",
    "safety_car_flag",
    "weather_impact_factor",
    "avg_pit_stops_circuit",
    "track_complexity",
    "qualifying_position",
    "grid_penalty",
]


def repo_relative(path):
    return path.relative_to(BASE_DIR).as_posix()


def validar_base(df, nome):
    erros = []

    for coluna in COLUNAS_OBRIGATORIAS:
        if coluna not in df.columns:
            erros.append(f"{nome}: coluna obrigatoria ausente: {coluna}")

    if erros:
        return erros

    if df.empty:
        erros.append(f"{nome}: base vazia")

    duplicatas = int(df.duplicated("RaceID").sum())
    if duplicatas:
        erros.append(f"{nome}: {duplicatas} RaceIDs duplicados")

    nulos_obrigatorios = df[COLUNAS_OBRIGATORIAS].isna().sum()
    nulos_obrigatorios = nulos_obrigatorios[nulos_obrigatorios > 0]
    if not nulos_obrigatorios.empty:
        erros.append(f"{nome}: nulos em colunas obrigatorias: {nulos_obrigatorios.to_dict()}")

    if df["season"].min() < 2018:
        erros.append(f"{nome}: contem temporada anterior ao recorte oficial 2018")

    if (df["grid_position"] <= 0).any():
        erros.append(f"{nome}: grid_position <= 0 apos correcao de pit lane")

    if (df["finish_position"] <= 0).any():
        erros.append(f"{nome}: finish_position <= 0")

    if "points" in df.columns:
        # points pode existir na entrada da etapa, mas nao na saida FE-ready.
        pass

    return erros


def preparar_base(df):
    df = df.copy()

    # Alias final esperado pela arquitetura. Mantem a coluna original tambem,
    # pois o pipeline atual ainda usa compound_ordinal em etapas anteriores.
    if "tire_compound_start" not in df.columns:
        df["tire_compound_start"] = df["compound_ordinal"]

    if "season_factor" not in df.columns:
        df["season_factor"] = df["season"].astype(int)

    df = adicionar_avg_pitstops_causal(df)

    remover = [c for c in COLUNAS_REMOVER_DA_BASE_FE if c in df.columns]
    df = df.drop(columns=remover)

    return df


def adicionar_avg_pitstops_causal(df):
    df = df.copy()

    if "avg_pit_stops_circuit" in df.columns:
        df["avg_pit_stops_circuit_static_global"] = df["avg_pit_stops_circuit"]

    if "fastf1_pit_in_count" not in df.columns:
        return df

    race_level = (
        df.groupby(["season", "round", "circuit_id"], as_index=False)
        .agg(race_avg_pit_stops=("fastf1_pit_in_count", "mean"))
        .sort_values(["season", "round", "circuit_id"])
        .reset_index(drop=True)
    )

    race_level["avg_pit_stops_circuit"] = (
        race_level.groupby("circuit_id")["race_avg_pit_stops"]
        .transform(lambda s: s.expanding().mean().shift(1))
    )

    race_level["avg_pit_stops_global_prior"] = (
        race_level["race_avg_pit_stops"].expanding().mean().shift(1)
    )

    race_level["avg_pit_stops_circuit_cold_start_flag"] = (
        race_level["avg_pit_stops_circuit"].isna()
    ).astype(int)

    race_level["avg_pit_stops_circuit"] = (
        race_level["avg_pit_stops_circuit"]
        .fillna(race_level["avg_pit_stops_global_prior"])
        .fillna(0.0)
    )

    causal = race_level[
        [
            "season",
            "round",
            "circuit_id",
            "avg_pit_stops_circuit",
            "avg_pit_stops_circuit_cold_start_flag",
        ]
    ]

    df = df.drop(
        columns=["avg_pit_stops_circuit", "avg_pit_stops_circuit_cold_start_flag"],
        errors="ignore",
    )
    df = df.merge(causal, on=["season", "round", "circuit_id"], how="left")
    return df


def salvar_target(df, path):
    target = df[COLUNAS_CHAVE + [TARGET]].copy()
    target.to_csv(path, index=False, encoding="utf-8-sig")


def salvar_outliers_revisao(df):
    if "outlier_tipo" not in df.columns:
        outliers = pd.DataFrame()
    else:
        outliers = df[df["outlier_tipo"] == "outlier_revisao"].copy()

    colunas_preferidas = [
        "season",
        "round",
        "race_name",
        "driver_id",
        "constructor_id",
        "RaceID",
        "grid_position",
        "finish_position",
        "status",
        "fastf1_avg_lap_time",
        "fastf1_best_lap_time",
        "fastf1_avg_sector1",
        "fastf1_avg_sector2",
        "fastf1_avg_sector3",
        "outlier_colunas",
        "safety_car_flag",
        "corrida_chuva_flag",
        "outlier_tipo",
    ]
    colunas = [c for c in colunas_preferidas if c in outliers.columns]
    outliers = outliers[colunas].sort_values(["season", "round", "driver_id"])
    outliers.to_csv(OUTLIERS_REVISAO, index=False, encoding="utf-8-sig")
    return outliers


def montar_manifest(df_2024, df_2025):
    colunas_saida = df_2025.columns.tolist()
    colunas_proibidas_presentes = [
        c for c in COLUNAS_PROIBIDAS_MODELO if c in colunas_saida
    ]
    colunas_removidas = [
        c for c in COLUNAS_REMOVER_DA_BASE_FE if c not in colunas_saida
    ]

    return {
        "recorte_temporal_oficial": "2018-2025",
        "entrada_oficial_feature_engineering": repo_relative(OUTPUT_FE_2018_2025),
        "entrada_treino_ate_2024": repo_relative(OUTPUT_FE_2018_2024),
        "target": TARGET,
        "target_files": {
            "2018_2024": repo_relative(TARGET_2018_2024),
            "2018_2025": repo_relative(TARGET_2018_2025),
        },
        "colunas_proibidas_modelo": COLUNAS_PROIBIDAS_MODELO,
        "colunas_proibidas_presentes_na_base_fe_ready": colunas_proibidas_presentes,
        "colunas_removidas_da_base_fe_ready": colunas_removidas,
        "observacao_target": (
            "finish_position permanece na base FE-ready para ser target e para "
            "calculos historicos causais. Deve ser removida de X antes da modelagem."
        ),
        "novas_colunas_de_preparacao": [
            "tire_compound_start",
            "season_factor",
            "avg_pit_stops_circuit_static_global",
            "avg_pit_stops_circuit_cold_start_flag",
        ],
        "validacoes": {
            "linhas_2018_2024": int(len(df_2024)),
            "linhas_2018_2025": int(len(df_2025)),
            "raceid_duplicados_2018_2024": int(df_2024.duplicated("RaceID").sum()),
            "raceid_duplicados_2018_2025": int(df_2025.duplicated("RaceID").sum()),
            "nulos_obrigatorios_2018_2024": int(df_2024[COLUNAS_OBRIGATORIAS].isna().sum().sum()),
            "nulos_obrigatorios_2018_2025": int(df_2025[COLUNAS_OBRIGATORIAS].isna().sum().sum()),
            "safety_car_corridas_2018_2024": int(
                df_2024[df_2024["safety_car_flag"] == 1]
                .groupby(["season", "round"])
                .ngroups
            ),
            "safety_car_corridas_2018_2025": int(
                df_2025[df_2025["safety_car_flag"] == 1]
                .groupby(["season", "round"])
                .ngroups
            ),
            "pitstop_cold_start_rows_2018_2025": int(
                df_2025["avg_pit_stops_circuit_cold_start_flag"].sum()
            ),
        },
    }


def salvar_relatorio(df_2024, df_2025, outliers, manifest, erros_entrada, erros_saida):
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("RELATORIO - 09 PREPARACAO PARA FEATURE ENGINEERING\n")
        f.write("=" * 70 + "\n\n")

        f.write("ENTRADAS\n")
        f.write("-" * 70 + "\n")
        f.write(f"{repo_relative(INPUT_2018_2024)}\n")
        f.write(f"{repo_relative(INPUT_2018_2025)}\n\n")

        f.write("SAIDAS\n")
        f.write("-" * 70 + "\n")
        f.write(f"{repo_relative(OUTPUT_FE_2018_2024)}\n")
        f.write(f"{repo_relative(OUTPUT_FE_2018_2025)}\n")
        f.write(f"{repo_relative(TARGET_2018_2024)}\n")
        f.write(f"{repo_relative(TARGET_2018_2025)}\n")
        f.write(f"{repo_relative(OUTLIERS_REVISAO)}\n")
        f.write(f"{repo_relative(MANIFEST_FILE)}\n\n")

        f.write("VALIDACAO\n")
        f.write("-" * 70 + "\n")
        if erros_entrada or erros_saida:
            f.write("ERROS ENCONTRADOS:\n")
            for erro in erros_entrada + erros_saida:
                f.write(f"- {erro}\n")
        else:
            f.write("Nenhum erro bloqueante encontrado.\n")

        f.write("\nANTI-LEAKAGE\n")
        f.write("-" * 70 + "\n")
        f.write(f"Target: {TARGET}\n")
        f.write("Colunas proibidas em X:\n")
        for coluna in COLUNAS_PROIBIDAS_MODELO:
            f.write(f"- {coluna}\n")
        f.write("\nColunas proibidas ainda presentes na base FE-ready:\n")
        for coluna in manifest["colunas_proibidas_presentes_na_base_fe_ready"]:
            f.write(f"- {coluna}\n")
        f.write("\nObservacao: finish_position permanece apenas como target/historico.\n")

        f.write("\nDIMENSOES\n")
        f.write("-" * 70 + "\n")
        f.write(f"2018-2024: {df_2024.shape}\n")
        f.write(f"2018-2025: {df_2025.shape}\n")

        f.write("\nSAFETY CAR\n")
        f.write("-" * 70 + "\n")
        f.write(
            "Corridas 2018-2024 com SC/VSC: "
            f"{manifest['validacoes']['safety_car_corridas_2018_2024']}\n"
        )
        f.write(
            "Corridas 2018-2025 com SC/VSC: "
            f"{manifest['validacoes']['safety_car_corridas_2018_2025']}\n"
        )

        f.write("\nOUTLIERS EM REVISAO\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total: {len(outliers)}\n")
        if not outliers.empty:
            f.write(outliers[["season", "round", "race_name", "driver_id", "outlier_colunas"]].to_string(index=False))
            f.write("\n")

        f.write("\nAVG PIT STOPS CAUSAL\n")
        f.write("-" * 70 + "\n")
        f.write(
            "`avg_pit_stops_circuit` foi recalculada usando apenas corridas "
            "anteriores do mesmo circuito. A media global antiga foi preservada "
            "em `avg_pit_stops_circuit_static_global` apenas para auditoria.\n"
        )
        f.write(
            "Linhas cold-start 2018-2025: "
            f"{manifest['validacoes']['pitstop_cold_start_rows_2018_2025']}\n"
        )


def salvar_documentacao():
    texto = """# Preparacao da base para Feature Engineering

## Recorte oficial

O recorte temporal oficial do projeto e 2018 em diante.

## Base oficial

A base oficial para a etapa de Feature Engineering e:

- `data/processed/dataset_feature_engineering_ready_2018_2025.csv`

Para experimentos que precisam treinar apenas ate 2024, usar:

- `data/processed/dataset_feature_engineering_ready_2018_2024.csv`

## Anti-leakage

O target do problema e `finish_position`.

As seguintes colunas nunca devem entrar em `X`:

- `finish_position`
- `points`
- `race_points`
- `fastest_lap_race`
- `previous_position`

`finish_position` permanece na base pronta para Feature Engineering porque e necessario como target e como historico para features causais, como `recent_form_3`, `recent_form_5`, vitorias acumuladas e sinergia piloto-construtor. Antes da modelagem, ele deve ser separado de `X`.

`points` foi removido da base pronta por ser uma variavel pos-corrida fortemente derivada do resultado.

## Pit stops sem vazamento temporal

A coluna `avg_pit_stops_circuit` foi recalculada na base pronta usando apenas corridas anteriores do mesmo circuito. A media global por circuito produzida na etapa 07 foi preservada como `avg_pit_stops_circuit_static_global` apenas para auditoria e nao deve ser usada como feature temporal principal.

## Arquivos gerados

- `dataset_feature_engineering_ready_2018_2024.csv`
- `dataset_feature_engineering_ready_2018_2025.csv`
- `target_finish_position_2018_2024.csv`
- `target_finish_position_2018_2025.csv`
- `outliers_revisao_2018_2025.csv`
- `manifest_feature_engineering.json`
- `relatorio_09_preparacao_feature_engineering.txt`
"""

    with open(DOC_FILE, "w", encoding="utf-8") as f:
        f.write(texto)


def main():
    print("Carregando bases pre-features...")
    df_2024_in = pd.read_csv(INPUT_2018_2024)
    df_2025_in = pd.read_csv(INPUT_2018_2025)

    erros_entrada = []
    erros_entrada.extend(validar_base(df_2024_in, INPUT_2018_2024.name))
    erros_entrada.extend(validar_base(df_2025_in, INPUT_2018_2025.name))

    print("Preparando bases FE-ready...")
    df_2024 = preparar_base(df_2024_in)
    df_2025 = preparar_base(df_2025_in)

    erros_saida = []
    erros_saida.extend(validar_base(df_2024, OUTPUT_FE_2018_2024.name))
    erros_saida.extend(validar_base(df_2025, OUTPUT_FE_2018_2025.name))

    if "points" in df_2024.columns or "points" in df_2025.columns:
        erros_saida.append("points ainda esta presente em base FE-ready")

    outliers = salvar_outliers_revisao(df_2025)
    salvar_target(df_2024, TARGET_2018_2024)
    salvar_target(df_2025, TARGET_2018_2025)

    df_2024.to_csv(OUTPUT_FE_2018_2024, index=False, encoding="utf-8-sig")
    df_2025.to_csv(OUTPUT_FE_2018_2025, index=False, encoding="utf-8-sig")

    manifest = montar_manifest(df_2024, df_2025)
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    salvar_relatorio(df_2024, df_2025, outliers, manifest, erros_entrada, erros_saida)
    salvar_documentacao()

    print(f"FE-ready 2018-2024: {df_2024.shape}")
    print(f"FE-ready 2018-2025: {df_2025.shape}")
    print(f"Outliers em revisao: {len(outliers)}")
    print(f"Manifest: {MANIFEST_FILE}")
    print(f"Relatorio: {REPORT_FILE}")

    if erros_entrada or erros_saida:
        print("\nErros encontrados:")
        for erro in erros_entrada + erros_saida:
            print(f"- {erro}")
        raise SystemExit(1)

    print("\nEtapa 09 finalizada com sucesso.")


if __name__ == "__main__":
    main()
