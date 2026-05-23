# EDA e Validacao do Dataset Tratado

Gerado em: 2026-05-21 10:27:16
Dataset: `data/processed/dataset_feature_engineering_ready_2018_2025.csv`
Status geral: **APROVADO**

## Resumo do dataset

- Linhas: 2943
- Colunas: 122
- Valores nulos: 0
- Linhas duplicadas: 0
- Cobertura temporal: 2018 a 2025
- Registros por temporada: `{2018: 335, 2019: 360, 2020: 283, 2021: 381, 2022: 366, 2023: 374, 2024: 425, 2025: 419}`
- Target `finish_position`: min=1, max=20, media=9.11
- Linhas com fallback cold-start em pit stops: 511
- Tipos de outlier: `{'nao_outlier': 2917, 'outlier_legitimo': 26}`

## Validacao de regras

- Regras aprovadas: 33
- Regras reprovadas: 0
- Resultado JSON: `reports/eda_dataset_tratado/great_expectations/checkpoint_result.json`
- Resumo Markdown: `reports/eda_dataset_tratado/great_expectations/validation_summary.md`
- Great Expectations Core: `{'status': 'ok', 'success': True, 'version': '1.6.4', 'context_type': 'EphemeralDataContext', 'result': 'reports/eda_dataset_tratado/great_expectations/gx_core_validation_result.json'}`

## Relatorios automaticos

- YData Profiling: `{'html': 'reports/eda_dataset_tratado/ydata_profile_dataset_tratado.html', 'json': 'reports/eda_dataset_tratado/ydata_profile_dataset_tratado.json'}`
- Sweetviz: `{'dataset': 'reports/eda_dataset_tratado/suplementar/sweetviz/sweetviz_dataset_tratado.html', 'train_vs_2025': 'reports/eda_dataset_tratado/suplementar/sweetviz/sweetviz_train_vs_2025.html', 'safety_car': 'reports/eda_dataset_tratado/suplementar/sweetviz/sweetviz_safety_car_vs_no_safety_car.html'}`

## Graficos gerados

- `reports/eda_dataset_tratado/figures/eda_principal/01_missing_values.png`
- `reports/eda_dataset_tratado/figures/eda_principal/02_target_distribution.png`
- `reports/eda_dataset_tratado/figures/eda_principal/03_target_by_season.png`
- `reports/eda_dataset_tratado/figures/eda_principal/04_grid_vs_finish.png`
- `reports/eda_dataset_tratado/figures/eda_principal/05_qualifying_vs_finish.png`
- `reports/eda_dataset_tratado/figures/eda_principal/06_correlation_heatmap.png`
- `reports/eda_dataset_tratado/figures/eda_principal/07_outlier_resumo.png`
- `reports/eda_dataset_tratado/figures/eda_principal/07b_outlier_distribuicoes_separadas.png`
- `reports/eda_dataset_tratado/figures/eda_principal/08_pit_stops_distribution.png`
- `reports/eda_dataset_tratado/figures/eda_principal/09_position_gain_distribution.png`
- `reports/eda_dataset_tratado/figures/eda_principal/10_weather_impact_distribution.png`
- `reports/eda_dataset_tratado/figures/eda_principal/11_track_complexity_distribution.png`
- `reports/eda_dataset_tratado/figures/eda_principal/12_temporal_summary.png`

## Decisao

O dataset fica aprovado para a proxima etapa quando todas as regras criticas passam e os relatorios automaticos nao indicam anomalias nao explicadas.
