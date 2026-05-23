# Validacao de Qualidade - Dataset Tratado

- Total de regras: 33
- Regras aprovadas: 33
- Regras reprovadas: 0

| Regra | Status | Observado | Detalhe |
|---|---:|---|---|
| `dataset_not_empty` | PASS | `2943` | Dataset deve ter linhas. |
| `critical_columns_present` | PASS | `[]` | Colunas criticas para EDA/pre-FE devem existir. |
| `no_null_values` | PASS | `0` | Dataset tratado nao deve ter nulos. |
| `no_full_duplicate_rows` | PASS | `0` | Sem linhas totalmente duplicadas. |
| `unique_driver_per_race` | PASS | `{"subset": ["season", "round", "driver_id"], "duplicates": 0}` | Cada piloto deve aparecer uma vez por corrida. |
| `raceid_unique` | PASS | `0` | RaceID deve ser unico por linha. |
| `season_within_range` | PASS | `{"invalid": 0, "min": 2018.0, "max": 2025.0}` | season deve estar entre 2018 e 2025. |
| `round_within_range` | PASS | `{"invalid": 0, "min": 1.0, "max": 24.0}` | round deve estar entre 1 e 30. |
| `grid_position_within_range` | PASS | `{"invalid": 0, "min": 1.0, "max": 21.0}` | grid_position deve estar entre 1 e 24. |
| `qualifying_position_within_range` | PASS | `{"invalid": 0, "min": 1.0, "max": 21.0}` | qualifying_position deve estar entre 1 e 24. |
| `finish_position_within_range` | PASS | `{"invalid": 0, "min": 1.0, "max": 20.0}` | finish_position deve estar entre 1 e 24. |
| `laps_within_range` | PASS | `{"invalid": 0, "min": 1.0, "max": 87.0}` | laps deve estar entre 0 e 200. |
| `compound_ordinal_within_range` | PASS | `{"invalid": 0, "min": 0.0, "max": 3.0}` | compound_ordinal deve estar entre 0 e 5. |
| `tire_compound_start_within_range` | PASS | `{"invalid": 0, "min": 0.0, "max": 3.0}` | tire_compound_start deve estar entre 0 e 5. |
| `safety_car_flag_within_range` | PASS | `{"invalid": 0, "min": 0.0, "max": 1.0}` | safety_car_flag deve estar entre 0 e 1. |
| `weather_impact_factor_within_range` | PASS | `{"invalid": 0, "min": 9.236722497973916e-12, "max": 0.9675567798674544}` | weather_impact_factor deve estar entre 0 e 1. |
| `avg_pit_stops_circuit_within_range` | PASS | `{"invalid": 0, "min": 0.0, "max": 3.80479302832244}` | avg_pit_stops_circuit deve estar entre 0 e 10. |
| `track_complexity_within_range` | PASS | `{"invalid": 0, "min": 0.1077283259612044, "max": 0.7447271451806697}` | track_complexity deve estar entre 0 e 1. |
| `altitude_m_within_range` | PASS | `{"invalid": 0, "min": 0.0, "max": 2285.0}` | altitude_m deve estar entre -100 e 2500. |
| `circuit_type_within_range` | PASS | `{"invalid": 0, "min": 0.0, "max": 1.0}` | circuit_type deve estar entre 0 e 5. |
| `grid_position_zero_flag_is_binary` | PASS | `[]` | grid_position_zero_flag deve ser flag binaria. |
| `wet_compound_flag_is_binary` | PASS | `[]` | wet_compound_flag deve ser flag binaria. |
| `corrida_chuva_flag_is_binary` | PASS | `[]` | corrida_chuva_flag deve ser flag binaria. |
| `outlier_flag_is_binary` | PASS | `[]` | outlier_flag deve ser flag binaria. |
| `dnf_flag_is_binary` | PASS | `[]` | dnf_flag deve ser flag binaria. |
| `dnf_driver_flag_is_binary` | PASS | `[]` | dnf_driver_flag deve ser flag binaria. |
| `dnf_car_flag_is_binary` | PASS | `[]` | dnf_car_flag deve ser flag binaria. |
| `dnf_other_flag_is_binary` | PASS | `[]` | dnf_other_flag deve ser flag binaria. |
| `safety_car_flag_is_binary` | PASS | `[]` | safety_car_flag deve ser flag binaria. |
| `avg_pit_stops_circuit_cold_start_flag_is_binary` | PASS | `[]` | avg_pit_stops_circuit_cold_start_flag deve ser flag binaria. |
| `season_minimum_volume` | PASS | `{}` | Cada temporada deve ter volume minimo plausivel de registros. |
| `race_grid_size_plausible` | PASS | `{}` | Cada corrida deve ter entre 10 e 24 registros de pilotos. |
| `target_has_variation` | PASS | `20` | Target precisa ter variacao. |
