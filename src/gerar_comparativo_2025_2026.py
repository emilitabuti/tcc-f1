from __future__ import annotations

import csv
import shutil
import subprocess
from html import escape
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
REPORTS_DIR = BASE_DIR / "reports" / "modelagem"
FIGURES_DIR = REPORTS_DIR / "figures" / "comparativos"

METRICAS_FOLDS = REPORTS_DIR / "tabela_metricas_tunadas_4modelos.csv"
METRICAS_2026_CORRIDA = REPORTS_DIR / "metricas_2026_por_corrida.csv"

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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def preparar_dados() -> tuple[list[str], dict[str, list[float]]]:
    folds = _read_csv(METRICAS_FOLDS)
    metricas_2026 = _read_csv(METRICAS_2026_CORRIDA)

    rounds = sorted({int(row["round"]) for row in metricas_2026})
    etapas = ["Fold 2025", *[f"R{round_} 2026" for round_ in rounds]]
    dados = {modelo: [0.0 for _ in etapas] for modelo in MODEL_ORDER}

    for row in folds:
        if int(row["valid_season"]) != 2025:
            continue
        modelo = MODEL_LABELS[row["modelo"]]
        dados[modelo][0] = float(row["mae"])

    round_index = {round_: index + 1 for index, round_ in enumerate(rounds)}
    for row in metricas_2026:
        modelo = row["modelo_label"]
        dados[modelo][round_index[int(row["round"])]] = float(row["mae"])

    return etapas, dados


def _polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def gerar_svg(etapas: list[str], dados: dict[str, list[float]]) -> str:
    width, height = 1180, 650
    margin_left, margin_right = 92, 260
    margin_top, margin_bottom = 82, 92
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom

    valores = [valor for serie in dados.values() for valor in serie if valor > 0]
    y_min = max(0.0, min(valores) - 0.25)
    y_max = max(valores) + 0.25
    y_step = 0.5
    ticks_y = []
    tick = round(y_min * 2) / 2
    while tick <= y_max + 0.001:
        if tick >= y_min - 0.001:
            ticks_y.append(tick)
        tick += y_step

    def x_pos(index: int) -> float:
        if len(etapas) == 1:
            return margin_left + chart_width / 2
        return margin_left + chart_width * index / (len(etapas) - 1)

    def y_pos(value: float) -> float:
        return margin_top + chart_height * (y_max - value) / (y_max - y_min)

    elements: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#263238}.title{font-size:24px;font-weight:700}.subtitle{font-size:14px;fill:#546e7a}.axis{font-size:13px}.legend{font-size:14px}.note{font-size:12px;fill:#607d8b}</style>',
        '<text class="title" x="92" y="42">Comparativo de MAE entre 2025 e 2026</text>',
        '<text class="subtitle" x="92" y="66">Fold temporal de 2025 versus corridas disponíveis da validação exploratória de 2026</text>',
    ]

    for tick in ticks_y:
        y = y_pos(tick)
        elements.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{margin_left + chart_width}" y2="{y:.1f}" stroke="#eceff1" stroke-width="1"/>')
        elements.append(f'<text class="axis" x="{margin_left - 14}" y="{y + 4:.1f}" text-anchor="end">{tick:.1f}</text>')

    elements.extend(
        [
            f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + chart_height}" stroke="#263238" stroke-width="1.2"/>',
            f'<line x1="{margin_left}" y1="{margin_top + chart_height}" x2="{margin_left + chart_width}" y2="{margin_top + chart_height}" stroke="#263238" stroke-width="1.2"/>',
            f'<line x1="{(x_pos(0) + x_pos(1)) / 2:.1f}" y1="{margin_top}" x2="{(x_pos(0) + x_pos(1)) / 2:.1f}" y2="{margin_top + chart_height}" stroke="#78909c" stroke-width="1.2" stroke-dasharray="6 6"/>',
            f'<text class="note" x="{x_pos(0):.1f}" y="{margin_top + 18}" text-anchor="middle">Validação histórica</text>',
            f'<text class="note" x="{(x_pos(1) + x_pos(len(etapas) - 1)) / 2:.1f}" y="{margin_top + 18}" text-anchor="middle">Validação exploratória</text>',
            f'<text class="axis" x="28" y="{margin_top + chart_height / 2:.1f}" transform="rotate(-90 28 {margin_top + chart_height / 2:.1f})" text-anchor="middle">MAE</text>',
        ]
    )

    for index, etapa in enumerate(etapas):
        x = x_pos(index)
        label = escape(etapa.replace(" ", "\n"))
        parts = label.split("\n")
        elements.append(f'<line x1="{x:.1f}" y1="{margin_top + chart_height}" x2="{x:.1f}" y2="{margin_top + chart_height + 6}" stroke="#263238"/>')
        for part_index, part in enumerate(parts):
            elements.append(
                f'<text class="axis" x="{x:.1f}" y="{margin_top + chart_height + 24 + part_index * 16}" text-anchor="middle">{part}</text>'
            )

    for modelo in MODEL_ORDER:
        points = [(x_pos(index), y_pos(valor)) for index, valor in enumerate(dados[modelo])]
        color = PALETTE[modelo]
        elements.append(f'<polyline points="{_polyline(points)}" fill="none" stroke="{color}" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>')
        for x, y in points:
            elements.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}" stroke="#ffffff" stroke-width="1.5"/>')

    legend_x = margin_left + chart_width + 44
    legend_y = margin_top + 42
    elements.append(f'<text class="legend" x="{legend_x}" y="{legend_y - 24}" font-weight="700">Modelo</text>')
    for index, modelo in enumerate(MODEL_ORDER):
        y = legend_y + index * 32
        color = PALETTE[modelo]
        elements.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 28}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        elements.append(f'<circle cx="{legend_x + 14}" cy="{y}" r="5" fill="{color}" stroke="#ffffff" stroke-width="1.5"/>')
        elements.append(f'<text class="legend" x="{legend_x + 40}" y="{y + 5}">{escape(modelo)}</text>')

    elements.append(
        f'<text class="note" x="{margin_left}" y="{height - 20}">Observação: 2025 representa o fold completo de validação; 2026 representa corridas individuais disponíveis.</text>'
    )
    elements.append("</svg>")
    return "\n".join(elements)


def main() -> None:
    etapas, dados = preparar_dados()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output = FIGURES_DIR / "mae_2025_2026_comparativo.svg"
    output.write_text(gerar_svg(etapas, dados), encoding="utf-8")
    chrome = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")
    if chrome:
        subprocess.run(
            [
                chrome,
                "--headless",
                "--disable-gpu",
                "--no-sandbox",
                f"--screenshot={output.with_suffix('.png')}",
                "--window-size=1180,650",
                str(output),
            ],
            check=True,
        )
    print(output.relative_to(BASE_DIR))


if __name__ == "__main__":
    main()
