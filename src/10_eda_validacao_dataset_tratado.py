from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except Exception:  # pragma: no cover - optional dependency fallback
    sns = None


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = BASE_DIR / "data" / "processed" / "dataset_feature_engineering_ready_2018_2025.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "reports" / "eda_dataset_tratado"

TARGET = "finish_position"
KEY_COLUMNS = ["season", "round", "RaceID", "driver_id", "constructor_id"]
CRITICAL_COLUMNS = [
    "season",
    "round",
    "RaceID",
    "driver_id",
    "constructor_id",
    "grid_position",
    "qualifying_position",
    TARGET,
    "laps",
    "compound_ordinal",
    "tire_compound_start",
    "safety_car_flag",
    "weather_impact_factor",
    "avg_pit_stops_circuit",
    "track_complexity",
    "altitude_m",
    "circuit_type",
]
FLAG_COLUMNS = [
    "grid_position_zero_flag",
    "wet_compound_flag",
    "corrida_chuva_flag",
    "outlier_flag",
    "dnf_flag",
    "dnf_driver_flag",
    "dnf_car_flag",
    "dnf_other_flag",
    "safety_car_flag",
    "avg_pit_stops_circuit_cold_start_flag",
]
RANGE_RULES = {
    "season": (2018, 2025),
    "round": (1, 30),
    "grid_position": (1, 24),
    "qualifying_position": (1, 24),
    TARGET: (1, 24),
    "laps": (0, 200),
    "compound_ordinal": (0, 5),
    "tire_compound_start": (0, 5),
    "safety_car_flag": (0, 1),
    "weather_impact_factor": (0, 1),
    "avg_pit_stops_circuit": (0, 10),
    "track_complexity": (0, 1),
    "altitude_m": (-100, 2500),
    "circuit_type": (0, 5),
}
PLOT_FEATURES = [
    "grid_position",
    "qualifying_position",
    "track_complexity",
    "weather_impact_factor",
    "avg_pit_stops_circuit",
    "altitude_m",
    "laps",
    "fastf1_avg_lap_time",
    "fastf1_best_lap_time",
]


@dataclass
class CheckResult:
    name: str
    success: bool
    observed: Any
    details: str


def repo_relative(path: Path) -> str:
    try:
        return path.relative_to(BASE_DIR).as_posix()
    except ValueError:
        return path.as_posix()


def ensure_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "root": output_dir,
        "figures": output_dir / "figures",
        "gx": output_dir / "great_expectations",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def setup_plot_style() -> None:
    if sns is not None:
        sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 160,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def available_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def run_ydata_profile(df: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    output_path = output_dir / "ydata_profile_dataset_tratado.html"
    json_path = output_dir / "ydata_profile_dataset_tratado.json"
    try:
        from ydata_profiling import ProfileReport
    except Exception as exc:
        return {"status": "skipped", "reason": f"{type(exc).__name__}: {exc}"}

    attempts = [
        (
            "full",
            {
                "explorative": True,
                "minimal": False,
                "correlations": {
                    "auto": {"calculate": True},
                    "pearson": {"calculate": True},
                    "spearman": {"calculate": True},
                },
            },
        ),
        (
            "minimal",
            {
                "minimal": True,
            },
        ),
    ]
    errors: list[str] = []
    for mode, kwargs in attempts:
        try:
            profile = ProfileReport(
                df,
                title=f"EDA Dataset Tratado F1 - Feature Engineering Ready ({mode})",
                **kwargs,
            )
            profile.to_file(output_path)
            try:
                profile.to_file(json_path)
            except Exception as exc:
                json_path.write_text(
                    json.dumps({"status": "skipped", "reason": f"{type(exc).__name__}: {exc}"}, indent=2),
                    encoding="utf-8",
                )
            return {
                "status": "ok",
                "mode": mode,
                "html": repo_relative(output_path),
                "json": repo_relative(json_path),
                "fallback_errors": errors,
            }
        except Exception as exc:
            errors.append(f"{mode}: {type(exc).__name__}: {exc}")

    return {"status": "skipped", "reason": " | ".join(errors)}


def run_sweetviz_reports(df: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    try:
        import sweetviz as sv
    except Exception as exc:
        return {"status": "skipped", "reason": f"{type(exc).__name__}: {exc}"}

    feature_config = sv.FeatureConfig(skip=[c for c in ["RaceID"] if c in df.columns])

    try:
        report = sv.analyze([df, "Dataset tratado"], target_feat=TARGET, feat_cfg=feature_config)
        path = output_dir / "sweetviz_dataset_tratado.html"
        report.show_html(filepath=str(path), open_browser=False, layout="widescreen")
        reports["dataset"] = repo_relative(path)
    except Exception as exc:
        reports["dataset"] = f"failed: {type(exc).__name__}: {exc}"

    if "season" in df.columns:
        train = df[df["season"] <= 2024]
        validation = df[df["season"] == 2025]
        if not train.empty and not validation.empty:
            try:
                report = sv.compare(
                    [train, "Historico 2018-2024"],
                    [validation, "Validacao 2025"],
                    target_feat=TARGET,
                    feat_cfg=feature_config,
                )
                path = output_dir / "sweetviz_train_vs_2025.html"
                report.show_html(filepath=str(path), open_browser=False, layout="widescreen")
                reports["train_vs_2025"] = repo_relative(path)
            except Exception as exc:
                reports["train_vs_2025"] = f"failed: {type(exc).__name__}: {exc}"

    if "safety_car_flag" in df.columns:
        try:
            report = sv.compare_intra(
                df,
                df["safety_car_flag"].astype(int) == 1,
                ["Com safety car", "Sem safety car"],
                target_feat=TARGET,
                feat_cfg=feature_config,
            )
            path = output_dir / "sweetviz_safety_car_vs_no_safety_car.html"
            report.show_html(filepath=str(path), open_browser=False, layout="widescreen")
            reports["safety_car"] = repo_relative(path)
        except Exception as exc:
            reports["safety_car"] = f"failed: {type(exc).__name__}: {exc}"

    reports["status"] = "ok"
    return reports


def add_check(results: list[CheckResult], name: str, success: bool, observed: Any, details: str) -> None:
    results.append(CheckResult(name=name, success=bool(success), observed=observed, details=details))


def validate_dataset(df: pd.DataFrame) -> list[CheckResult]:
    results: list[CheckResult] = []

    add_check(results, "dataset_not_empty", len(df) > 0, len(df), "Dataset deve ter linhas.")

    missing_columns = [col for col in CRITICAL_COLUMNS if col not in df.columns]
    add_check(
        results,
        "critical_columns_present",
        not missing_columns,
        missing_columns,
        "Colunas criticas para EDA/pre-FE devem existir.",
    )

    total_nulls = int(df.isna().sum().sum())
    add_check(results, "no_null_values", total_nulls == 0, total_nulls, "Dataset tratado nao deve ter nulos.")

    duplicate_rows = int(df.duplicated().sum())
    add_check(results, "no_full_duplicate_rows", duplicate_rows == 0, duplicate_rows, "Sem linhas totalmente duplicadas.")

    key_subset = available_columns(df, ["season", "round", "driver_id"])
    if key_subset:
        duplicate_keys = int(df.duplicated(key_subset).sum())
        add_check(
            results,
            "unique_driver_per_race",
            duplicate_keys == 0,
            {"subset": key_subset, "duplicates": duplicate_keys},
            "Cada piloto deve aparecer uma vez por corrida.",
        )

    if "RaceID" in df.columns:
        duplicate_raceid = int(df["RaceID"].duplicated().sum())
        add_check(results, "raceid_unique", duplicate_raceid == 0, duplicate_raceid, "RaceID deve ser unico por linha.")

    for col, (min_value, max_value) in RANGE_RULES.items():
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        invalid = int((series.lt(min_value) | series.gt(max_value) | series.isna()).sum())
        add_check(
            results,
            f"{col}_within_range",
            invalid == 0,
            {"invalid": invalid, "min": float(series.min()), "max": float(series.max())},
            f"{col} deve estar entre {min_value} e {max_value}.",
        )

    for col in available_columns(df, FLAG_COLUMNS):
        values = set(pd.Series(df[col]).dropna().unique().tolist())
        invalid_values = sorted(v for v in values if v not in {0, 1, False, True})
        add_check(
            results,
            f"{col}_is_binary",
            not invalid_values,
            invalid_values,
            f"{col} deve ser flag binaria.",
        )

    if "season" in df.columns and TARGET in df.columns:
        counts = df.groupby("season")[TARGET].count().to_dict()
        sparse = {int(k): int(v) for k, v in counts.items() if v < 100}
        add_check(
            results,
            "season_minimum_volume",
            not sparse,
            sparse,
            "Cada temporada deve ter volume minimo plausivel de registros.",
        )

    if "RaceID" in df.columns and "season" in df.columns and "round" in df.columns:
        pilots_per_race = df.groupby(["season", "round"])["RaceID"].count()
        invalid_races = pilots_per_race[(pilots_per_race < 10) | (pilots_per_race > 24)]
        add_check(
            results,
            "race_grid_size_plausible",
            invalid_races.empty,
            {str(k): int(v) for k, v in invalid_races.to_dict().items()},
            "Cada corrida deve ter entre 10 e 24 registros de pilotos.",
        )

    if TARGET in df.columns:
        target_unique = int(df[TARGET].nunique())
        add_check(
            results,
            "target_has_variation",
            target_unique > 1,
            target_unique,
            "Target precisa ter variacao.",
        )

    return results


def write_validation_outputs(results: list[CheckResult], output_dir: Path) -> dict[str, Any]:
    gx_dir = output_dir / "great_expectations"
    serializable = [
        {
            "expectation": result.name,
            "success": result.success,
            "observed": result.observed,
            "details": result.details,
        }
        for result in results
    ]
    result_path = gx_dir / "checkpoint_result.json"
    result_path.write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")

    passed = sum(1 for item in results if item.success)
    failed = len(results) - passed
    lines = [
        "# Validacao de Qualidade - Dataset Tratado",
        "",
        f"- Total de regras: {len(results)}",
        f"- Regras aprovadas: {passed}",
        f"- Regras reprovadas: {failed}",
        "",
        "| Regra | Status | Observado | Detalhe |",
        "|---|---:|---|---|",
    ]
    for result in results:
        status = "PASS" if result.success else "FAIL"
        observed = json.dumps(result.observed, ensure_ascii=False, default=str)
        lines.append(f"| `{result.name}` | {status} | `{observed}` | {result.details} |")
    summary_path = gx_dir / "validation_summary.md"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "status": "ok" if failed == 0 else "failed",
        "passed": passed,
        "failed": failed,
        "json": repo_relative(result_path),
        "summary": repo_relative(summary_path),
    }


def run_great_expectations_core(df: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    """Best-effort GX Core validation using the public pandas API.

    The custom contract above stays as the concise Markdown/JSON output used by
    the TCC. This function also emits a native GX validation result when GX Core
    is installed, so the run is auditable from the framework itself.
    """
    try:
        import great_expectations as gx
        import great_expectations.expectations as gxe

        context = gx.get_context(mode="ephemeral")
        suite = gx.ExpectationSuite(name="dataset_tratado_quality_suite")

        suite.add_expectation(gxe.ExpectTableRowCountToBeBetween(min_value=1))
        suite.add_expectation(
            gxe.ExpectTableColumnsToMatchSet(
                column_set=[col for col in CRITICAL_COLUMNS if col in df.columns],
                exact_match=False,
            )
        )

        for col in available_columns(df, CRITICAL_COLUMNS):
            suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column=col))

        if "RaceID" in df.columns:
            suite.add_expectation(gxe.ExpectColumnValuesToBeUnique(column="RaceID"))

        for col, (min_value, max_value) in RANGE_RULES.items():
            if col in df.columns:
                suite.add_expectation(
                    gxe.ExpectColumnValuesToBeBetween(
                        column=col,
                        min_value=min_value,
                        max_value=max_value,
                    )
                )

        for col in available_columns(df, FLAG_COLUMNS):
            suite.add_expectation(gxe.ExpectColumnValuesToBeInSet(column=col, value_set=[0, 1]))

        context.suites.add(suite)
        datasource = context.data_sources.add_pandas("dataset_tratado_pandas")
        asset = datasource.add_dataframe_asset(name="dataset_feature_engineering_ready")
        batch_definition = asset.add_batch_definition_whole_dataframe("full_dataset")
        batch = batch_definition.get_batch(batch_parameters={"dataframe": df})
        result = batch.validate(suite)

        output_path = output_dir / "great_expectations" / "gx_core_validation_result.json"
        output_path.write_text(json.dumps(result.to_json_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

        return {
            "status": "ok",
            "success": bool(result.success),
            "version": getattr(gx, "__version__", "unknown"),
            "context_type": type(context).__name__,
            "result": repo_relative(output_path),
        }
    except Exception as exc:
        return {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}


def plot_missing_values(df: pd.DataFrame, path: Path) -> None:
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    plt.figure(figsize=(10, 4))
    if missing.empty:
        plt.bar(["sem_nulos"], [0], color="#2c7fb8")
        plt.ylabel("Valores ausentes")
        plt.title("Valores ausentes por coluna")
    else:
        missing.head(40).plot(kind="bar", color="#2c7fb8")
        plt.ylabel("Valores ausentes")
        plt.title("Valores ausentes por coluna - top 40")
        plt.xticks(rotation=75, ha="right")
    savefig(path)


def plot_target_distribution(df: pd.DataFrame, path: Path) -> None:
    if TARGET not in df.columns:
        return
    plt.figure(figsize=(8, 4.5))
    if sns is not None:
        sns.histplot(df[TARGET], bins=range(int(df[TARGET].min()), int(df[TARGET].max()) + 2), kde=False, color="#2c7fb8")
    else:
        plt.hist(df[TARGET], bins=20, color="#2c7fb8")
    plt.title("Distribuicao do target finish_position")
    plt.xlabel("finish_position")
    plt.ylabel("Frequencia")
    savefig(path)


def plot_target_by_season(df: pd.DataFrame, path: Path) -> None:
    if TARGET not in df.columns or "season" not in df.columns:
        return
    plt.figure(figsize=(9, 4.8))
    if sns is not None:
        sns.boxplot(data=df, x="season", y=TARGET, color="#8dd3c7")
    else:
        seasons = sorted(df["season"].unique())
        plt.boxplot([df.loc[df["season"] == season, TARGET] for season in seasons], labels=seasons)
    plt.title("finish_position por temporada")
    plt.xlabel("Temporada")
    plt.ylabel("finish_position")
    savefig(path)


def plot_scatter_with_target(df: pd.DataFrame, feature: str, path: Path) -> None:
    if TARGET not in df.columns or feature not in df.columns:
        return
    sample = df[[feature, TARGET]].dropna()
    plt.figure(figsize=(7, 5))
    if sns is not None:
        sns.regplot(data=sample, x=feature, y=TARGET, scatter_kws={"alpha": 0.35, "s": 18}, line_kws={"color": "#d95f02"})
    else:
        plt.scatter(sample[feature], sample[TARGET], alpha=0.35, s=18)
    plt.title(f"{feature} vs finish_position")
    savefig(path)


def plot_correlation_heatmap(df: pd.DataFrame, path: Path) -> None:
    preferred = available_columns(df, [TARGET] + PLOT_FEATURES)
    if len(preferred) < 2:
        return
    corr = df[preferred].corr(numeric_only=True)
    plt.figure(figsize=(9, 7))
    if sns is not None:
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, square=True, linewidths=0.5)
    else:
        plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar()
        plt.xticks(range(len(corr)), corr.columns, rotation=75, ha="right")
        plt.yticks(range(len(corr)), corr.index)
    plt.title("Correlacao entre target e features numericas principais")
    savefig(path)


def plot_outlier_boxplots(df: pd.DataFrame, path: Path) -> None:
    columns = available_columns(df, PLOT_FEATURES)
    if not columns:
        return
    n = len(columns)
    rows = int(np.ceil(n / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(13, 3.5 * rows))
    axes = np.array(axes).reshape(-1)
    for ax, col in zip(axes, columns):
        if sns is not None:
            sns.boxplot(y=df[col], ax=ax, color="#bebada")
        else:
            ax.boxplot(df[col].dropna(), vert=True)
        ax.set_title(col)
        ax.set_xlabel("")
    for ax in axes[len(columns) :]:
        ax.axis("off")
    fig.suptitle("Boxplots para inspecao de outliers", y=1.01)
    savefig(path)


def plot_feature_distribution(df: pd.DataFrame, feature: str, path: Path) -> None:
    if feature not in df.columns:
        return
    plt.figure(figsize=(8, 4.5))
    if sns is not None:
        sns.histplot(df[feature], kde=True, color="#66a61e")
    else:
        plt.hist(df[feature].dropna(), bins=25, color="#66a61e")
    plt.title(f"Distribuicao de {feature}")
    savefig(path)


def plot_categorical_balance(df: pd.DataFrame, path: Path) -> None:
    candidates = available_columns(df, ["compound_normalizado", "dnf_categoria", "circuit_type", "tire_compound_start"])
    if not candidates:
        return
    fig, axes = plt.subplots(len(candidates), 1, figsize=(10, 3.5 * len(candidates)))
    axes = np.array([axes]).reshape(-1)
    for ax, col in zip(axes, candidates):
        counts = df[col].value_counts().head(20)
        if sns is not None:
            sns.barplot(x=counts.values, y=counts.index.astype(str), ax=ax, color="#80b1d3")
        else:
            ax.barh(counts.index.astype(str), counts.values, color="#80b1d3")
        ax.set_title(f"Distribuicao de {col}")
        ax.set_xlabel("Registros")
        ax.set_ylabel("")
    savefig(path)


def plot_temporal_summary(df: pd.DataFrame, path: Path) -> None:
    metrics = available_columns(df, [TARGET, "weather_impact_factor", "track_complexity", "avg_pit_stops_circuit"])
    if "season" not in df.columns or not metrics:
        return
    summary = df.groupby("season")[metrics].mean().reset_index()
    plt.figure(figsize=(10, 5))
    for metric in metrics:
        plt.plot(summary["season"], summary[metric], marker="o", label=metric)
    plt.title("Media anual de metricas principais")
    plt.xlabel("Temporada")
    plt.legend()
    savefig(path)


def generate_figures(df: pd.DataFrame, figures_dir: Path) -> list[str]:
    setup_plot_style()
    outputs = [
        ("01_missing_values.png", lambda p: plot_missing_values(df, p)),
        ("02_target_distribution.png", lambda p: plot_target_distribution(df, p)),
        ("03_target_by_season.png", lambda p: plot_target_by_season(df, p)),
        ("04_grid_vs_finish.png", lambda p: plot_scatter_with_target(df, "grid_position", p)),
        ("05_qualifying_vs_finish.png", lambda p: plot_scatter_with_target(df, "qualifying_position", p)),
        ("06_correlation_heatmap.png", lambda p: plot_correlation_heatmap(df, p)),
        ("07_outlier_boxplots.png", lambda p: plot_outlier_boxplots(df, p)),
        ("08_weather_impact_distribution.png", lambda p: plot_feature_distribution(df, "weather_impact_factor", p)),
        ("09_track_complexity_distribution.png", lambda p: plot_feature_distribution(df, "track_complexity", p)),
        ("10_pit_stops_distribution.png", lambda p: plot_feature_distribution(df, "avg_pit_stops_circuit", p)),
        ("11_categorical_balance.png", lambda p: plot_categorical_balance(df, p)),
        ("12_temporal_summary.png", lambda p: plot_temporal_summary(df, p)),
    ]
    created: list[str] = []
    for filename, plotter in outputs:
        path = figures_dir / filename
        try:
            plotter(path)
            if path.exists():
                created.append(repo_relative(path))
        except Exception as exc:
            print(f"[WARN] Falha ao gerar {filename}: {type(exc).__name__}: {exc}")
            plt.close("all")
    return created


def dataset_summary(df: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "null_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }
    if "season" in df.columns:
        summary["season_min"] = int(df["season"].min())
        summary["season_max"] = int(df["season"].max())
        summary["rows_by_season"] = {int(k): int(v) for k, v in df["season"].value_counts().sort_index().items()}
    if TARGET in df.columns:
        summary["target_min"] = float(df[TARGET].min())
        summary["target_max"] = float(df[TARGET].max())
        summary["target_mean"] = float(df[TARGET].mean())
    if "outlier_tipo" in df.columns:
        summary["outlier_tipo"] = {str(k): int(v) for k, v in df["outlier_tipo"].value_counts(dropna=False).items()}
    if "avg_pit_stops_circuit_cold_start_flag" in df.columns:
        summary["avg_pit_stops_cold_start_rows"] = int(df["avg_pit_stops_circuit_cold_start_flag"].sum())
    return summary


def write_main_summary(
    output_dir: Path,
    input_path: Path,
    df: pd.DataFrame,
    summary: dict[str, Any],
    validation: dict[str, Any],
    gx_core: dict[str, Any],
    ydata_result: dict[str, Any],
    sweetviz_result: dict[str, Any],
    figures: list[str],
) -> Path:
    status = "APROVADO" if validation["failed"] == 0 else "REVISAR"
    lines = [
        "# EDA e Validacao do Dataset Tratado",
        "",
        f"Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Dataset: `{repo_relative(input_path)}`",
        f"Status geral: **{status}**",
        "",
        "## Resumo do dataset",
        "",
        f"- Linhas: {summary['rows']}",
        f"- Colunas: {summary['columns']}",
        f"- Valores nulos: {summary['null_values']}",
        f"- Linhas duplicadas: {summary['duplicate_rows']}",
    ]
    if "season_min" in summary:
        lines.extend(
            [
                f"- Cobertura temporal: {summary['season_min']} a {summary['season_max']}",
                f"- Registros por temporada: `{summary['rows_by_season']}`",
            ]
        )
    if "target_min" in summary:
        lines.extend(
            [
                f"- Target `{TARGET}`: min={summary['target_min']:.0f}, max={summary['target_max']:.0f}, media={summary['target_mean']:.2f}",
            ]
        )
    if "avg_pit_stops_cold_start_rows" in summary:
        lines.append(f"- Linhas com fallback cold-start em pit stops: {summary['avg_pit_stops_cold_start_rows']}")
    if "outlier_tipo" in summary:
        lines.append(f"- Tipos de outlier: `{summary['outlier_tipo']}`")

    lines.extend(
        [
            "",
            "## Validacao de regras",
            "",
            f"- Regras aprovadas: {validation['passed']}",
            f"- Regras reprovadas: {validation['failed']}",
            f"- Resultado JSON: `{validation['json']}`",
            f"- Resumo Markdown: `{validation['summary']}`",
            f"- Great Expectations Core: `{gx_core}`",
            "",
            "## Relatorios automaticos",
            "",
            f"- YData Profiling: `{ydata_result}`",
            f"- Sweetviz: `{sweetviz_result}`",
            "",
            "## Graficos gerados",
            "",
        ]
    )
    lines.extend(f"- `{figure}`" for figure in figures)
    lines.extend(
        [
            "",
            "## Decisao",
            "",
            "O dataset fica aprovado para a proxima etapa quando todas as regras criticas passam e os relatorios automaticos nao indicam anomalias nao explicadas.",
        ]
    )
    path = output_dir / "eda_dataset_tratado_summary.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gera EDA, validacoes e graficos para o dataset tratado.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="CSV tratado a validar.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Diretorio dos relatorios.")
    parser.add_argument("--skip-heavy-reports", action="store_true", help="Pula YData Profiling e Sweetviz.")
    parser.add_argument("--skip-ydata", action="store_true", help="Pula o relatorio YData Profiling.")
    parser.add_argument("--skip-sweetviz", action="store_true", help="Pula os relatorios Sweetviz.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    dirs = ensure_dirs(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Dataset nao encontrado: {input_path}")

    df = pd.read_csv(input_path)
    summary = dataset_summary(df)

    validation_results = validate_dataset(df)
    validation = write_validation_outputs(validation_results, dirs["root"])
    gx_core = run_great_expectations_core(df, dirs["root"])

    figures = generate_figures(df, dirs["figures"])

    if args.skip_heavy_reports or args.skip_ydata:
        existing_ydata = dirs["root"] / "ydata_profile_dataset_tratado.html"
        existing_ydata_json = dirs["root"] / "ydata_profile_dataset_tratado.json"
        ydata_result = (
            {
                "status": "existing",
                "html": repo_relative(existing_ydata),
                "json": repo_relative(existing_ydata_json),
            }
            if existing_ydata.exists()
            else {"status": "skipped", "reason": "--skip-ydata or --skip-heavy-reports"}
        )
    else:
        ydata_result = run_ydata_profile(df, dirs["root"])

    if args.skip_heavy_reports or args.skip_sweetviz:
        existing = {
            "dataset": dirs["root"] / "sweetviz_dataset_tratado.html",
            "train_vs_2025": dirs["root"] / "sweetviz_train_vs_2025.html",
            "safety_car": dirs["root"] / "sweetviz_safety_car_vs_no_safety_car.html",
        }
        sweetviz_result = {
            key: repo_relative(path) for key, path in existing.items() if path.exists()
        }
        sweetviz_result["status"] = "existing" if len(sweetviz_result) > 1 else "skipped"
        if sweetviz_result["status"] == "skipped":
            sweetviz_result["reason"] = "--skip-sweetviz or --skip-heavy-reports"
    else:
        sweetviz_result = run_sweetviz_reports(df, dirs["root"])

    summary_path = write_main_summary(
        dirs["root"],
        input_path,
        df,
        summary,
        validation,
        gx_core,
        ydata_result,
        sweetviz_result,
        figures,
    )

    print(f"EDA concluido: {repo_relative(summary_path)}")
    print(f"Validacao: {validation['passed']} regras aprovadas, {validation['failed']} reprovadas")
    return 0 if validation["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
