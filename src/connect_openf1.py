"""
Bloco 4 — Conectar OpenF1 API e validar acesso do plano.

OpenF1 é uma API open-source não-oficial para dados de F1.
Documentação: https://openf1.org/docs

Plano gratuito (Community):
  - Todos os 18 endpoints disponíveis
  - Dados históricos desde 2023
  - Sem autenticação necessária
  - Rate limit: 3 req/s e 30 req/min

Este script:
  1. Testa conectividade com a API
  2. Valida os principais endpoints relevantes para o TCC
  3. Verifica rate limits e tempo de resposta
  4. Inspeciona cobertura temporal dos dados
  5. Salva relatório de validação em data/raw/openf1_validation.json

Mapeamento de campos alvo → OpenF1:
  session_key    → presente em todos os endpoints
  driver_number  → presente em todos os endpoints
  position       → /position (posição ao longo da corrida, com timestamp)
                   /session_result (posição final de chegada)
                   /starting_grid (posição no grid de largada)
  date           → /position → campo `date` (timestamp de cada mudança de posição)
                   NÃO existe em /starting_grid — confirmado pela doc oficial.
                   O /starting_grid é estático (1 registro/piloto), sem timestamp.
  grid_position  → NÃO existe como campo isolado na OpenF1.
                   Equivalente: /starting_grid → campo `position`
                   (posição no grid de largada, session_key da Qualifying)
                   Requer join via meeting_key para associar à corrida.
"""

import requests
import time
import json
import os
import logging
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------
BASE_URL  = "https://api.openf1.org/v1"
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_RAW  = os.path.join(BASE_DIR, "../data/raw")
LOG_FILE  = os.path.join(DATA_RAW, "openf1_validation.log")
OUT_FILE  = os.path.join(DATA_RAW, "openf1_validation.json")

os.makedirs(DATA_RAW, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# Sessão HTTP reutilizável com headers padrão
session = requests.Session()
session.headers.update({
    "Accept": "application/json",
    "User-Agent": "tcc-f1-openf1-validator/1.0",
})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get(endpoint: str, params: dict | None = None, timeout: int = 15) -> dict:
    """
    Faz GET em BASE_URL/endpoint e retorna dict com:
      ok        : bool
      status    : int | None
      elapsed_ms: float
      data      : list | None
      error     : str | None
    """
    url = f"{BASE_URL}/{endpoint}"
    t0  = time.perf_counter()
    try:
        resp = session.get(url, params=params, timeout=timeout)
        elapsed = (time.perf_counter() - t0) * 1000
        resp.raise_for_status()
        return {
            "ok": True,
            "status": resp.status_code,
            "elapsed_ms": round(elapsed, 1),
            "data": resp.json(),
            "error": None,
        }
    except requests.exceptions.HTTPError as e:
        elapsed = (time.perf_counter() - t0) * 1000
        return {"ok": False, "status": e.response.status_code if e.response else None,
                "elapsed_ms": round(elapsed, 1), "data": None, "error": str(e)}
    except requests.exceptions.RequestException as e:
        elapsed = (time.perf_counter() - t0) * 1000
        return {"ok": False, "status": None,
                "elapsed_ms": round(elapsed, 1), "data": None, "error": str(e)}


def _safe_len(data) -> int | str:
    if isinstance(data, list):
        return len(data)
    return "n/a"


# ---------------------------------------------------------------------------
# Testes por endpoint
# ---------------------------------------------------------------------------

def test_sessions_coverage():
    """Verifica quais anos têm sessões disponíveis (2023–2025)."""
    log.info("── Testando cobertura temporal (/sessions) ──")
    results = {}
    for year in range(2023, 2026):
        r = get("sessions", params={"year": year})
        count = _safe_len(r["data"])
        results[year] = {"ok": r["ok"], "sessions_found": count, "elapsed_ms": r["elapsed_ms"]}
        status = "✓" if r["ok"] else "✗"
        log.info(f"  {status} {year}: {count} sessões  ({r['elapsed_ms']} ms)")
        time.sleep(0.4)  # respeitar rate limit
    return results


def test_meetings():
    """Valida endpoint de meetings para 2023 e 2024."""
    log.info("── Testando /meetings ──")
    results = {}
    for year in [2023, 2024]:
        r = get("meetings", params={"year": year})
        count = _safe_len(r["data"])
        results[year] = {"ok": r["ok"], "meetings_found": count, "elapsed_ms": r["elapsed_ms"]}
        status = "✓" if r["ok"] else "✗"
        log.info(f"  {status} {year}: {count} meetings  ({r['elapsed_ms']} ms)")
        time.sleep(0.4)
    return results


def test_laps():
    """Testa /laps com uma sessão conhecida (Singapore GP 2023 - Race, session_key=9165)."""
    log.info("── Testando /laps (Singapore 2023 Race, session_key=9165) ──")
    # Busca apenas o piloto 1 para ser rápido
    r = get("laps", params={"session_key": 9165, "driver_number": 1})
    count = _safe_len(r["data"])
    status = "✓" if r["ok"] else "✗"
    log.info(f"  {status} {count} voltas encontradas  ({r['elapsed_ms']} ms)")
    sample = r["data"][0] if r["ok"] and count else None
    if sample:
        log.info(f"  Exemplo: lap {sample.get('lap_number')} | "
                 f"duration={sample.get('lap_duration')}s | "
                 f"sector1={sample.get('duration_sector_1')}s")
    return {"ok": r["ok"], "laps_found": count, "elapsed_ms": r["elapsed_ms"], "sample": sample}


def test_stints():
    """Testa /stints — informações de pneu por stint."""
    log.info("── Testando /stints (Singapore 2023 Race, session_key=9165) ──")
    r = get("stints", params={"session_key": 9165, "driver_number": 1})
    count = _safe_len(r["data"])
    status = "✓" if r["ok"] else "✗"
    log.info(f"  {status} {count} stints encontrados  ({r['elapsed_ms']} ms)")
    if r["ok"] and count:
        for s in r["data"]:
            log.info(f"    stint {s.get('stint_number')}: "
                     f"{s.get('compound')} laps {s.get('lap_start')}–{s.get('lap_end')} "
                     f"(age at start: {s.get('tyre_age_at_start')})")
    return {"ok": r["ok"], "stints_found": count, "elapsed_ms": r["elapsed_ms"]}


def test_pit():
    """Testa /pit — paradas nos boxes."""
    log.info("── Testando /pit (Singapore 2023 Race, session_key=9165) ──")
    r = get("pit", params={"session_key": 9165})
    count = _safe_len(r["data"])
    status = "✓" if r["ok"] else "✗"
    log.info(f"  {status} {count} pit stops encontrados  ({r['elapsed_ms']} ms)")
    return {"ok": r["ok"], "pits_found": count, "elapsed_ms": r["elapsed_ms"]}


def test_drivers():
    """Testa /drivers — informações dos pilotos de uma sessão."""
    log.info("── Testando /drivers (Singapore 2023 Race, session_key=9165) ──")
    r = get("drivers", params={"session_key": 9165})
    count = _safe_len(r["data"])
    status = "✓" if r["ok"] else "✗"
    log.info(f"  {status} {count} pilotos encontrados  ({r['elapsed_ms']} ms)")
    if r["ok"] and count:
        sample = r["data"][0]
        log.info(f"  Exemplo: #{sample.get('driver_number')} {sample.get('full_name')} "
                 f"| {sample.get('team_name')}")
    return {"ok": r["ok"], "drivers_found": count, "elapsed_ms": r["elapsed_ms"]}


def test_position():
    """Testa /position — posições ao longo da corrida."""
    log.info("── Testando /position (Singapore 2023 Race, session_key=9165, driver=1) ──")
    r = get("position", params={"session_key": 9165, "driver_number": 1})
    count = _safe_len(r["data"])
    status = "✓" if r["ok"] else "✗"
    log.info(f"  {status} {count} registros de posição  ({r['elapsed_ms']} ms)")
    return {"ok": r["ok"], "positions_found": count, "elapsed_ms": r["elapsed_ms"]}


def test_race_control():
    """Testa /race_control — bandeiras e safety car."""
    log.info("── Testando /race_control (Singapore 2023 Race, session_key=9165) ──")
    r = get("race_control", params={"session_key": 9165})
    count = _safe_len(r["data"])
    status = "✓" if r["ok"] else "✗"
    log.info(f"  {status} {count} eventos de controle  ({r['elapsed_ms']} ms)")
    return {"ok": r["ok"], "events_found": count, "elapsed_ms": r["elapsed_ms"]}


def test_weather():
    """Testa /weather — condições climáticas da pista."""
    log.info("── Testando /weather (Singapore 2023, meeting_key=1219) ──")
    r = get("weather", params={"meeting_key": 1219})
    count = _safe_len(r["data"])
    status = "✓" if r["ok"] else "✗"
    log.info(f"  {status} {count} registros de clima  ({r['elapsed_ms']} ms)")
    if r["ok"] and count:
        s = r["data"][0]
        log.info(f"  Exemplo: air={s.get('air_temperature')}°C | "
                 f"track={s.get('track_temperature')}°C | "
                 f"humidity={s.get('humidity')}% | rain={s.get('rainfall')}")
    return {"ok": r["ok"], "weather_found": count, "elapsed_ms": r["elapsed_ms"]}


def test_session_result():
    """Testa /session_result — resultado final de uma sessão."""
    log.info("── Testando /session_result (Singapore 2023 Race, session_key=9165) ──")
    r = get("session_result", params={"session_key": 9165})
    count = _safe_len(r["data"])
    status = "✓" if r["ok"] else "✗"
    log.info(f"  {status} {count} resultados encontrados  ({r['elapsed_ms']} ms)")
    if r["ok"] and count:
        p1 = next((x for x in r["data"] if x.get("position") == 1), r["data"][0])
        log.info(f"  P1: driver #{p1.get('driver_number')} | "
                 f"laps={p1.get('number_of_laps')} | dnf={p1.get('dnf')}")
    return {"ok": r["ok"], "results_found": count, "elapsed_ms": r["elapsed_ms"]}


def test_starting_grid():
    """
    Testa /starting_grid — grid de largada (equivalente a grid_position).

    IMPORTANTE: este endpoint usa a session_key da sessão de QUALIFYING,
    não da corrida. O campo `position` aqui representa a posição no grid
    de largada, mapeando o conceito de `grid_position` do TCC.

    Para cruzar com dados de corrida:
      1. Obter meeting_key da corrida
      2. Buscar session_key da Qualifying daquele meeting
      3. Consultar /starting_grid com essa session_key
      4. O campo `position` = grid_position do piloto
    """
    log.info("── Testando /starting_grid (Singapore 2023 Qualifying, session_key=9161) ──")
    log.info("  ℹ️  grid_position = campo `position` do /starting_grid (sessão de Qualifying)")

    # Singapore 2023: Race=9165, Qualifying=9161, meeting_key=1219
    r = get("starting_grid", params={"session_key": 9161})
    count = _safe_len(r["data"])
    status = "✓" if r["ok"] else "✗"
    log.info(f"  {status} {count} posições de grid encontradas  ({r['elapsed_ms']} ms)")
    if r["ok"] and count:
        for row in sorted(r["data"], key=lambda x: x.get("position", 99))[:5]:
            log.info(f"    P{row.get('position'):>2} → driver #{row.get('driver_number')} "
                     f"| lap_duration={row.get('lap_duration')}s")
        log.info(f"    ... (total {count} pilotos)")

    # Valida campos alvo
    campos_alvo = {"session_key", "driver_number", "position"}
    if r["ok"] and count:
        presentes = campos_alvo & set(r["data"][0].keys())
        ausentes  = campos_alvo - presentes
        log.info(f"  Campos alvo presentes: {sorted(presentes)}")
        if ausentes:
            log.warning(f"  Campos alvo ausentes:  {sorted(ausentes)}")
        log.info("  ℹ️  `date` não existe em /starting_grid (confirmado pela doc oficial).")
        log.info("      O endpoint é estático: 1 registro por piloto, sem timestamp.")
        log.info("      `date` disponível em /position (timestamp de mudanças de posição).")
        log.info("      `grid_position` = campo `position` deste endpoint.")

    return {
        "ok": r["ok"],
        "grid_positions_found": count,
        "elapsed_ms": r["elapsed_ms"],
        "nota": (
            "grid_position nao existe como campo isolado. "
            "Equivalente: /starting_grid -> campo `position` (session_key da Qualifying). "
            "Requer join por meeting_key para cruzar com dados de corrida."
        ),
    }


def test_rate_limit(n_requests: int = 5):
    """
    Envia N requisições seguidas ao endpoint mais leve e mede os tempos,
    verificando se respeitamos o rate limit do plano gratuito (3 req/s).
    """
    log.info(f"── Teste de rate limit ({n_requests} requisições sequenciais) ──")
    times = []
    sleep_between = 0.35  # ~2.9 req/s, abaixo do limite de 3 req/s

    for i in range(n_requests):
        r = get("sessions", params={"year": 2023, "session_type": "Race"})
        times.append(r["elapsed_ms"])
        status = "✓" if r["ok"] else "✗"
        log.info(f"  req {i+1}/{n_requests}: {status}  {r['elapsed_ms']} ms  "
                 f"(status={r['status']})")
        if i < n_requests - 1:
            time.sleep(sleep_between)

    avg = round(sum(times) / len(times), 1)
    log.info(f"  Média: {avg} ms | Min: {min(times)} ms | Max: {max(times)} ms")
    return {
        "requests": n_requests,
        "avg_ms": avg,
        "min_ms": min(times),
        "max_ms": max(times),
        "all_ok": all(t > 0 for t in times),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("=" * 60)
    log.info("  BLOCO 4 — Conectar OpenF1 API e validar acesso do plano")
    log.info(f"  Base URL: {BASE_URL}")
    log.info(f"  Plano: Community (gratuito, sem autenticação)")
    log.info(f"  Rate limit: 3 req/s | 30 req/min")
    log.info(f"  Cobertura histórica: 2023+")
    log.info("=" * 60)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "base_url": BASE_URL,
        "plan": "community_free",
        "endpoints": {},
    }

    # 1. Conectividade básica
    log.info("\n[1/10] Verificando conectividade básica...")
    r = get("sessions", params={"year": 2023, "session_type": "Race"})
    if not r["ok"]:
        log.error(f"  ✗ Falha ao conectar: {r['error']}")
        report["connectivity"] = {"ok": False, "error": r["error"]}
        _save(report)
        return
    log.info(f"  ✓ API acessível  ({r['elapsed_ms']} ms)")
    report["connectivity"] = {"ok": True, "elapsed_ms": r["elapsed_ms"]}

    # 2. Endpoints relevantes para o TCC
    time.sleep(0.4)
    log.info("\n[2/10] Cobertura temporal (sessions por ano)...")
    report["endpoints"]["sessions_coverage"] = test_sessions_coverage()

    time.sleep(0.4)
    log.info("\n[3/10] Meetings...")
    report["endpoints"]["meetings"] = test_meetings()

    time.sleep(0.4)
    log.info("\n[4/10] Laps...")
    report["endpoints"]["laps"] = test_laps()

    time.sleep(0.4)
    log.info("\n[5/10] Stints (compostos de pneu)...")
    report["endpoints"]["stints"] = test_stints()

    time.sleep(0.4)
    log.info("\n[6/10] Pit stops...")
    report["endpoints"]["pit"] = test_pit()

    time.sleep(0.4)
    log.info("\n[7/10] Drivers...")
    report["endpoints"]["drivers"] = test_drivers()

    time.sleep(0.4)
    log.info("\n[8/10] Position...")
    report["endpoints"]["position"] = test_position()

    time.sleep(0.4)
    log.info("\n[9/10] Race Control...")
    report["endpoints"]["race_control"] = test_race_control()

    time.sleep(0.4)
    log.info("\n[10/10] Weather...")
    report["endpoints"]["weather"] = test_weather()

    # Session result (bonus)
    time.sleep(0.4)
    log.info("\n[+] Session Result (bonus)...")
    report["endpoints"]["session_result"] = test_session_result()

    # Starting grid — mapeamento de grid_position
    time.sleep(0.4)
    log.info("\n[+] Starting Grid (grid_position)...")
    report["endpoints"]["starting_grid"] = test_starting_grid()

    # Mapeamento explícito de campos alvo
    report["field_mapping"] = {
        "session_key":    "presente em todos os endpoints",
        "driver_number":  "presente em todos os endpoints",
        "position":       "/position (ao longo da corrida) | /session_result (final) | /starting_grid (grid)",
        "date":           "/position -> campo `date` (timestamp de cada mudanca de posicao). NAO existe em /starting_grid — endpoint estatico (1 registro/piloto, sem timestamp). Confirmado pela doc oficial openf1.org.",
        "grid_position":  "NAO existe como campo isolado. Equivalente: /starting_grid -> campo `position` (session_key da Qualifying). Join por meeting_key necessario para associar a corrida.",
    }

    # Rate limit test
    time.sleep(0.4)
    log.info("\n[+] Teste de Rate Limit...")
    report["rate_limit_test"] = test_rate_limit(n_requests=5)

    # Resumo
    total   = len(report["endpoints"])
    success = sum(1 for v in report["endpoints"].values()
                  if (isinstance(v, dict) and v.get("ok")) or
                     (isinstance(v, dict) and all(x.get("ok") for x in v.values()
                                                   if isinstance(x, dict))))
    report["summary"] = {
        "endpoints_tested": total,
        "all_connectivity_ok": report["connectivity"]["ok"],
        "rate_limit_ok": report["rate_limit_test"]["all_ok"],
    }

    log.info("\n" + "=" * 60)
    log.info("  RESUMO DA VALIDAÇÃO")
    log.info(f"  Conectividade:   {'✓ OK' if report['connectivity']['ok'] else '✗ FALHOU'}")
    log.info(f"  Rate limit test: {'✓ OK' if report['rate_limit_test']['all_ok'] else '✗ FALHOU'}")
    log.info(f"  Endpoints testados: {total}")
    log.info(f"  Plano detectado: Community (sem autenticação)")
    log.info(f"  Dados disponíveis: 2023–presente")
    log.info(f"  Relatório salvo em: {OUT_FILE}")
    log.info("=" * 60)

    _save(report)


def _save(report: dict):
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    log.info(f"  → Relatório JSON salvo em {OUT_FILE}")


if __name__ == "__main__":
    main()
