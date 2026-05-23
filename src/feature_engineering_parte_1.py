from pathlib import Path
import argparse
import numpy as np
import pandas as pd


# 11 - Feature Engineering Parte 1
#
# Correções aplicadas:
# - merge RAPM protegido contra duplicação;
# - merge preferencial por RaceID quando disponível;
# - DNF rates calculadas causalmente no histórico classificado de DNF;
# - recent_form_3 e recent_form_5 com fallback 0 para cold-start;
# - validação para impedir que a base final saia com mais linhas que a entrada;
# - suporte para rodar 2018-2024 ou 2018-2025 via parâmetro.


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"

DEFAULT_INPUT_FILE = PROCESSED_DIR / "dataset_feature_engineering_ready_2018_2025.csv"
DEFAULT_DNF_FILE = PROCESSED_DIR / "historico_dnf_classificado_2018_2025.csv"

COEF_PILOTOS_FILE = PROCESSED_DIR / "coef_pilotos.csv"
COEF_CONSTRUTORES_FILE = PROCESSED_DIR / "coef_construtores.csv"

DEFAULT_OUTPUT_FILE = PROCESSED_DIR / "dataset_features_final_2018_2025.csv"
DEFAULT_REPORT_FILE = PROCESSED_DIR / "relatorio_feature_engineering.txt"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gera Feature Engineering Parte 1 com features causais."
    )

    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_FILE),
        help="Arquivo FE-ready de entrada.",
    )

    parser.add_argument(
        "--dnf-file",
        default=str(DEFAULT_DNF_FILE),
        help="Histórico classificado de DNFs.",
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_FILE),
        help="Arquivo CSV final de saída.",
    )

    parser.add_argument(
        "--report",
        default=str(DEFAULT_REPORT_FILE),
        help="Relatório de validação da etapa.",
    )

    return parser.parse_args()


def resolver_path(path_str):
    path = Path(path_str)

    if not path.is_absolute():
        path = BASE_DIR / path

    return path


def garantir_raceid(df):
    df = df.copy()

    if "RaceID" not in df.columns:
        df["RaceID"] = (
            df["driver_id"].astype(str)
            + "_"
            + df["season"].astype(str)
            + "_"
            + df["round"].astype(str)
        )

    return df


def validar_colunas(df):
    obrigatorias = [
        "season",
        "round",
        "race_name",
        "driver_id",
        "constructor_id",
        "finish_position",
    ]

    faltando = [col for col in obrigatorias if col not in df.columns]

    if faltando:
        raise ValueError(f"Colunas obrigatorias ausentes na base: {faltando}")


def criar_race_order(df):
    df = df.copy()

    if "race_order" in df.columns:
        return df

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
        validate="many_to_one",
    )

    return df


def carregar_base(input_file):
    if not input_file.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {input_file}")

    df = pd.read_csv(input_file)

    validar_colunas(df)

    df["season"] = df["season"].astype(int)
    df["round"] = df["round"].astype(int)
    df["finish_position"] = pd.to_numeric(df["finish_position"], errors="coerce")

    df = garantir_raceid(df)
    df = criar_race_order(df)

    df = df.sort_values(
        ["season", "round", "finish_position", "driver_id"]
    ).reset_index(drop=True)

    return df


def adicionar_race_order_temporal(df):
    df = df.copy()

    race_cols = ["season", "round"]
    if "race_name" in df.columns:
        race_cols.append("race_name")

    race_order = (
        df[race_cols]
        .drop_duplicates()
        .sort_values(["season", "round"])
        .reset_index(drop=True)
    )
    race_order["race_order"] = np.arange(1, len(race_order) + 1)

    df = df.drop(columns=["race_order"], errors="ignore")
    df = df.merge(race_order, on=race_cols, how="left", validate="many_to_one")

    return df


def preparar_coeficientes_rapm(coef_df, entidade_col, coef_nome):
    coef_df = coef_df.copy()

    if "coef_rapm" in coef_df.columns:
        coef_df = coef_df.rename(columns={"coef_rapm": coef_nome})

    if coef_nome not in coef_df.columns:
        raise ValueError(f"Coluna de coeficiente ausente: {coef_nome}")

    if "RaceID" in coef_df.columns:
        chaves = ["RaceID"]
    else:
        chaves = ["season", "round", entidade_col]

    colunas = chaves + [coef_nome]
    coef_df = coef_df[colunas].copy()

    # Correção principal:
    # garante que o lado direito do merge tenha somente uma linha por chave.
    # Isso impede multiplicar linhas da base final.
    coef_df = (
        coef_df
        .groupby(chaves, as_index=False)[coef_nome]
        .mean()
    )

    return coef_df, chaves


def adicionar_coeficientes_rapm(df):
    df = df.copy()
    linhas_antes = len(df)

    if not COEF_PILOTOS_FILE.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {COEF_PILOTOS_FILE}")

    if not COEF_CONSTRUTORES_FILE.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado: {COEF_CONSTRUTORES_FILE}")

    coef_pilotos = pd.read_csv(COEF_PILOTOS_FILE)
    coef_construtores = pd.read_csv(COEF_CONSTRUTORES_FILE)

    coef_pilotos, chaves_piloto = preparar_coeficientes_rapm(
        coef_pilotos,
        entidade_col="driver_id",
        coef_nome="driver_coef_rapm",
    )

    coef_construtores, chaves_construtor = preparar_coeficientes_rapm(
        coef_construtores,
        entidade_col="constructor_id",
        coef_nome="constructor_coef_rapm",
    )

    df = df.merge(
        coef_pilotos,
        on=chaves_piloto,
        how="left",
        validate="many_to_one",
    )

    df = df.merge(
        coef_construtores,
        on=chaves_construtor,
        how="left",
        validate="many_to_one",
    )

    linhas_depois = len(df)

    if linhas_depois != linhas_antes:
        raise RuntimeError(
            f"Erro no merge RAPM: linhas antes={linhas_antes}, linhas depois={linhas_depois}. "
            "O merge ainda esta duplicando registros."
        )

    df["driver_coef_rapm"] = df["driver_coef_rapm"].fillna(0)
    df["constructor_coef_rapm"] = df["constructor_coef_rapm"].fillna(0)

    return df


def weighted_recent_form(historico, n_corridas):
    if len(historico) == 0:
        return 0

    recentes = list(historico[-n_corridas:])

    # Inverte para o mais recente receber maior peso.
    # Exemplo para 5: corrida mais recente peso 5, depois 4, 3, 2, 1.
    recentes = recentes[::-1]

    pesos = np.arange(n_corridas, 0, -1, dtype=float)
    pesos = pesos[:len(recentes)]

    return float(np.average(recentes, weights=pesos))


def adicionar_recent_form(df):
    df = df.copy()

    df["recent_form_5"] = 0.0
    df["recent_form_3"] = 0.0
    df["recent_form_cold_start_flag"] = 0

    for driver_id, grupo in df.groupby("driver_id"):
        grupo = grupo.sort_values("race_order")

        historico = []

        for idx, row in grupo.iterrows():
            if len(historico) == 0:
                df.loc[idx, "recent_form_cold_start_flag"] = 1

            df.loc[idx, "recent_form_5"] = weighted_recent_form(
                historico,
                n_corridas=5,
            )

            df.loc[idx, "recent_form_3"] = weighted_recent_form(
                historico,
                n_corridas=3,
            )

            historico.append(row["finish_position"])

    return df


def adicionar_experiencia_e_vitorias(df):
    df = df.copy()

    df = df.sort_values(["driver_id", "race_order"])

    # Total de corridas anteriores do piloto.
    df["driver_experience"] = df.groupby("driver_id").cumcount()

    # Vitória da linha atual.
    df["driver_win_flag"] = (df["finish_position"] == 1).astype(int)

    # Vitórias anteriores do piloto.
    df["driver_wins_total"] = (
        df.groupby("driver_id")["driver_win_flag"]
        .transform(lambda s: s.cumsum().shift(1).fillna(0))
    )

    # Vitórias anteriores do construtor sem vazamento dentro da mesma corrida.
    constructor_race = (
        df.groupby(["constructor_id", "season", "round", "race_order"], as_index=False)
        .agg(constructor_win_race=("driver_win_flag", "max"))
        .sort_values(["constructor_id", "race_order"])
    )

    constructor_race["constructor_wins_total"] = (
        constructor_race.groupby("constructor_id")["constructor_win_race"]
        .transform(lambda s: s.cumsum().shift(1).fillna(0))
    )

    df = df.merge(
        constructor_race[
            [
                "constructor_id",
                "season",
                "round",
                "constructor_wins_total",
            ]
        ],
        on=["constructor_id", "season", "round"],
        how="left",
        validate="many_to_one",
    )

    df["driver_wins_total"] = df["driver_wins_total"].fillna(0)
    df["constructor_wins_total"] = df["constructor_wins_total"].fillna(0)

    return df


def normalizar_colunas_dnf(dnf):
    dnf = dnf.copy()

    if "raceId" in dnf.columns and "RaceID" not in dnf.columns:
        dnf = dnf.rename(columns={"raceId": "RaceID"})

    if "driver" in dnf.columns and "driver_id" not in dnf.columns:
        dnf = dnf.rename(columns={"driver": "driver_id"})

    if "constructor" in dnf.columns and "constructor_id" not in dnf.columns:
        dnf = dnf.rename(columns={"constructor": "constructor_id"})

    if "constructorId" in dnf.columns and "constructor_id" not in dnf.columns:
        dnf = dnf.rename(columns={"constructorId": "constructor_id"})

    if "is_dnf" not in dnf.columns:
        if "dnf_flag" in dnf.columns:
            dnf["is_dnf"] = dnf["dnf_flag"]
        else:
            dnf["is_dnf"] = 0

    if "dnf_driver_flag" not in dnf.columns:
        if "dnf_categoria" in dnf.columns:
            categoria = dnf["dnf_categoria"].astype(str).str.lower()
            dnf["dnf_driver_flag"] = categoria.str.contains(
                "piloto|driver|collision|accident|spun",
                regex=True,
            ).astype(int)
        else:
            dnf["dnf_driver_flag"] = 0

    if "dnf_car_flag" not in dnf.columns:
        if "dnf_categoria" in dnf.columns:
            categoria = dnf["dnf_categoria"].astype(str).str.lower()
            dnf["dnf_car_flag"] = categoria.str.contains(
                "carro|mecan|engine|gearbox|ers|hydraulic|brake|power|electrical",
                regex=True,
            ).astype(int)
        else:
            dnf["dnf_car_flag"] = 0

    dnf["is_dnf"] = pd.to_numeric(dnf["is_dnf"], errors="coerce").fillna(0).astype(int)
    dnf["dnf_driver_flag"] = pd.to_numeric(dnf["dnf_driver_flag"], errors="coerce").fillna(0).astype(int)
    dnf["dnf_car_flag"] = pd.to_numeric(dnf["dnf_car_flag"], errors="coerce").fillna(0).astype(int)

    return dnf


def carregar_historico_dnf(dnf_file):
    if not dnf_file.exists():
        raise FileNotFoundError(f"Arquivo de histórico DNF nao encontrado: {dnf_file}")

    dnf = pd.read_csv(dnf_file)
    dnf = normalizar_colunas_dnf(dnf)

    obrigatorias = [
        "season",
        "round",
        "driver_id",
        "constructor_id",
        "is_dnf",
        "dnf_driver_flag",
        "dnf_car_flag",
    ]

    faltando = [col for col in obrigatorias if col not in dnf.columns]

    if faltando:
        raise ValueError(f"Colunas obrigatorias ausentes no histórico DNF: {faltando}")

    dnf["season"] = dnf["season"].astype(int)
    dnf["round"] = dnf["round"].astype(int)

    dnf = garantir_raceid(dnf)
    dnf = adicionar_race_order_temporal(dnf)

    dnf = (
        dnf[
            [
                "RaceID",
                "season",
                "round",
                "race_order",
                "driver_id",
                "constructor_id",
                "is_dnf",
                "dnf_driver_flag",
                "dnf_car_flag",
            ]
        ]
        .drop_duplicates(subset=["RaceID"])
        .copy()
    )

    return dnf


def calcular_dnf_rates_historico(dnf):
    dnf = dnf.copy()

    dnf = dnf.sort_values(["driver_id", "race_order", "RaceID"])

    driver_starts_before = dnf.groupby("driver_id").cumcount()
    driver_dnf_before = (
        dnf.groupby("driver_id")["dnf_driver_flag"]
        .transform(lambda s: s.shift(1).fillna(0).cumsum())
    )

    dnf["driver_dnf_rate"] = np.where(
        driver_starts_before > 0,
        driver_dnf_before / driver_starts_before,
        0.0,
    )

    constructor_race = (
        dnf.groupby(["constructor_id", "season", "round", "race_order"], as_index=False)
        .agg(
            constructor_dnf_car_race=("dnf_car_flag", "sum"),
            constructor_entries_race=("RaceID", "count"),
        )
        .sort_values(["constructor_id", "race_order"])
    )

    constructor_dnf_before = (
        constructor_race.groupby("constructor_id")["constructor_dnf_car_race"]
        .transform(lambda s: s.shift(1).fillna(0).cumsum())
    )
    constructor_entries_before = (
        constructor_race.groupby("constructor_id")["constructor_entries_race"]
        .transform(lambda s: s.shift(1).fillna(0).cumsum())
    )

    constructor_race["constructor_dnf_rate"] = np.where(
        constructor_entries_before > 0,
        constructor_dnf_before / constructor_entries_before,
        0.0,
    )

    dnf = dnf.merge(
        constructor_race[
            [
                "constructor_id",
                "season",
                "round",
                "constructor_dnf_rate",
            ]
        ],
        on=["constructor_id", "season", "round"],
        how="left",
        validate="many_to_one",
    )

    dnf["driver_dnf_rate"] = dnf["driver_dnf_rate"].fillna(0)
    dnf["constructor_dnf_rate"] = dnf["constructor_dnf_rate"].fillna(0)

    return dnf


def adicionar_dnf_rates(df, dnf_file):
    df = df.copy()
    linhas_antes = len(df)

    dnf = carregar_historico_dnf(dnf_file)
    dnf = calcular_dnf_rates_historico(dnf)

    # Remove colunas DNF antigas da FE-ready para evitar conflito no merge.
    colunas_dnf_antigas = [
        "is_dnf",
        "dnf_flag",
        "dnf_driver_flag",
        "dnf_car_flag",
        "dnf_other_flag",
        "dnf_piloto_flag",
        "dnf_mecanico_flag",
        "dnf_geral_flag",
    ]

    colunas_para_remover = [col for col in colunas_dnf_antigas if col in df.columns]

    if colunas_para_remover:
        df = df.drop(columns=colunas_para_remover)

    dnf_rates = dnf[
        [
            "RaceID",
            "driver_id",
            "constructor_id",
            "driver_dnf_rate",
            "constructor_dnf_rate",
        ]
    ].copy()

    # Garante uma linha por piloto-corrida no historico de DNF.
    dnf_rates = (
        dnf_rates
        .groupby(["RaceID", "driver_id", "constructor_id"], as_index=False)
        .agg(
            driver_dnf_rate=("driver_dnf_rate", "max"),
            constructor_dnf_rate=("constructor_dnf_rate", "max"),
        )
    )

    df = df.merge(
        dnf_rates,
        on=["RaceID", "driver_id", "constructor_id"],
        how="left",
        validate="one_to_one",
    )

    linhas_depois = len(df)

    if linhas_depois != linhas_antes:
        raise RuntimeError(
            f"Erro no merge DNF: linhas antes={linhas_antes}, linhas depois={linhas_depois}."
        )

    df["driver_dnf_rate"] = df["driver_dnf_rate"].fillna(0)
    df["constructor_dnf_rate"] = df["constructor_dnf_rate"].fillna(0)

    return df


def adicionar_sinergia_piloto_construtor(df):
    df = df.copy()

    df["performance_score"] = -df["finish_position"]

    df = df.sort_values(["driver_id", "constructor_id", "race_order"])

    df["driver_constructor_synergy"] = (
        df.groupby(["driver_id", "constructor_id"])["performance_score"]
        .transform(lambda s: s.expanding().mean().shift(1))
    )

    df["driver_constructor_synergy"] = df["driver_constructor_synergy"].fillna(0)

    return df


def preservar_features_existentes(df):
    features_esperadas = [
        "track_complexity",
        "weather_impact_factor",
        "avg_pit_stops_circuit",
    ]

    existentes = [col for col in features_esperadas if col in df.columns]
    ausentes = [col for col in features_esperadas if col not in df.columns]

    return existentes, ausentes


def gerar_relatorio(df, input_file, output_file, report_file, existentes, ausentes, linhas_entrada):
    features_criadas = [
        "driver_coef_rapm",
        "constructor_coef_rapm",
        "recent_form_5",
        "recent_form_3",
        "recent_form_cold_start_flag",
        "driver_experience",
        "driver_wins_total",
        "driver_dnf_rate",
        "constructor_dnf_rate",
        "constructor_wins_total",
        "driver_constructor_synergy",
    ]

    raceid_duplicados = int(df["RaceID"].duplicated().sum()) if "RaceID" in df.columns else -1

    linhas = []

    linhas.append("Relatorio 11 - Feature Engineering Parte 1")
    linhas.append("=" * 45)
    linhas.append("")
    linhas.append(f"Arquivo de entrada: {input_file}")
    linhas.append(f"Arquivo de saida: {output_file}")
    linhas.append(f"Linhas de entrada: {linhas_entrada}")
    linhas.append(f"Linhas de saida: {len(df)}")
    linhas.append(f"RaceID duplicados na saida: {raceid_duplicados}")
    linhas.append(f"Temporadas: {int(df['season'].min())}-{int(df['season'].max())}")
    linhas.append("")
    linhas.append("Features criadas:")
    for feature in features_criadas:
        linhas.append(f"- {feature}")

    linhas.append("")
    linhas.append("Features ja existentes preservadas:")
    for feature in existentes:
        linhas.append(f"- {feature}")

    linhas.append("")
    linhas.append("Features esperadas nao encontradas na base:")
    for feature in ausentes:
        linhas.append(f"- {feature}")

    linhas.append("")
    linhas.append("Validacoes:")
    linhas.append(f"- driver_dnf_rate nonzero: {int((df['driver_dnf_rate'] != 0).sum())}")
    linhas.append(f"- constructor_dnf_rate nonzero: {int((df['constructor_dnf_rate'] != 0).sum())}")
    linhas.append(f"- recent_form_3 nulos: {int(df['recent_form_3'].isna().sum())}")
    linhas.append(f"- recent_form_5 nulos: {int(df['recent_form_5'].isna().sum())}")
    linhas.append(f"- driver_coef_rapm nulos: {int(df['driver_coef_rapm'].isna().sum())}")
    linhas.append(f"- constructor_coef_rapm nulos: {int(df['constructor_coef_rapm'].isna().sum())}")
    linhas.append(f"- features finais com nulos: {int(df[features_criadas].isna().sum().sum())}")
    linhas.append(f"- driver_dnf_rate minimo/maximo: {df['driver_dnf_rate'].min():.6f}/{df['driver_dnf_rate'].max():.6f}")
    linhas.append(f"- constructor_dnf_rate minimo/maximo: {df['constructor_dnf_rate'].min():.6f}/{df['constructor_dnf_rate'].max():.6f}")
    linhas.append("")
    linhas.append("Regra causal:")
    linhas.append("Todas as features historicas usam apenas dados anteriores a corrida atual.")
    linhas.append("DNF rates foram calculadas no historico DNF classificado antes do merge para a base DNF Excluded.")
    linhas.append("recent_form_3 e recent_form_5 usam media ponderada de finish_position, conforme a arquitetura.")
    linhas.append("driver_constructor_synergy usa -finish_position; maior valor indica melhor sinergia historica.")

    report_file.write_text("\n".join(linhas), encoding="utf-8")


def main():
    args = parse_args()

    input_file = resolver_path(args.input)
    dnf_file = resolver_path(args.dnf_file)
    output_file = resolver_path(args.output)
    report_file = resolver_path(args.report)

    df = carregar_base(input_file)
    linhas_entrada = len(df)

    existentes, ausentes = preservar_features_existentes(df)

    df = adicionar_coeficientes_rapm(df)
    df = adicionar_recent_form(df)
    df = adicionar_experiencia_e_vitorias(df)
    df = adicionar_dnf_rates(df, dnf_file)
    df = adicionar_sinergia_piloto_construtor(df)

    if "performance_score" in df.columns:
        df = df.drop(columns=["performance_score"])

    df = df.sort_values(
        ["season", "round", "finish_position", "driver_id"]
    ).reset_index(drop=True)

    if len(df) != linhas_entrada:
        raise RuntimeError(
            f"A base final alterou a quantidade de linhas. "
            f"Entrada={linhas_entrada}, saida={len(df)}."
        )

    if "RaceID" in df.columns and df["RaceID"].duplicated().sum() > 0:
        raise RuntimeError(
            f"Foram encontrados {df['RaceID'].duplicated().sum()} RaceID duplicados na saida."
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_file, index=False)

    gerar_relatorio(
        df=df,
        input_file=input_file,
        output_file=output_file,
        report_file=report_file,
        existentes=existentes,
        ausentes=ausentes,
        linhas_entrada=linhas_entrada,
    )

    print("Feature Engineering Parte 1 concluida com sucesso.")
    print(f"Linhas entrada: {linhas_entrada}")
    print(f"Linhas saida: {len(df)}")
    print(f"Arquivo gerado: {output_file}")
    print(f"Relatorio gerado: {report_file}")


if __name__ == "__main__":
    main()
