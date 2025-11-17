#!/usr/bin/env python3
"""Filter Transjakarta GTFS so Mikrotrans (JAK.*) data is separated.

This script now performs three operations:
1. Split `routes.csv` into bus (`routes_bus_only.csv`) and mikrotrans (`routes_mikrotrans.csv`).
2. Rewrite `stop_times.csv` so the root dataset only contains bus trips while
    dumping Mikrotrans trips to `filtered/mikrotrans/stop_times.csv`.
3. Rewrite `stops.csv` similarly, keeping its bus portion in-place and writing
    Mikrotrans stops (including shared stops) to `filtered/mikrotrans/stops.csv`.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
TRANSJAKARTA_DIR = ROOT / "data" / "transjakarta"
ROUTES_PATH = TRANSJAKARTA_DIR / "routes.csv"
BUS_ROUTES_PATH = TRANSJAKARTA_DIR / "routes_bus_only.csv"
MIKRO_ROUTES_PATH = TRANSJAKARTA_DIR / "routes_mikrotrans.csv"

TRIPS_PATH = TRANSJAKARTA_DIR / "trips.csv"
STOP_TIMES_PATH = TRANSJAKARTA_DIR / "stop_times.csv"
STOPS_PATH = TRANSJAKARTA_DIR / "stops.csv"

FILTERED_MIKRO_DIR = TRANSJAKARTA_DIR / "filtered" / "mikrotrans"
MIKRO_STOP_TIMES_PATH = FILTERED_MIKRO_DIR / "stop_times.csv"
MIKRO_STOPS_PATH = FILTERED_MIKRO_DIR / "stops.csv"


def _clean_cells(cells: Iterable[str]) -> List[str]:
    """Strip whitespace from each cell and return as list."""
    return [cell.strip() for cell in cells]


def _is_mikrotrans(route_id: str) -> bool:
    """Return True if the route id belongs to Mikrotrans (JAK.*)."""
    normalized = route_id.strip().upper()
    return normalized.startswith("JAK.")


def _read_csv(path: Path) -> Tuple[list[str], list[list[str]]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.reader(source)
        try:
            raw_header = next(reader)
        except StopIteration as exc:
            raise RuntimeError(f"{path.name} is empty") from exc

        header = _clean_cells(raw_header)
        num_columns = len(header)

        rows: list[list[str]] = []
        for line_number, raw_row in enumerate(reader, start=2):
            row = _clean_cells(raw_row)
            if not any(row):
                continue
            if len(row) != num_columns:
                raise ValueError(
                    "Column count mismatch",
                    {
                        "file": str(path),
                        "line": line_number,
                        "expected": num_columns,
                        "actual": len(row),
                        "row": row,
                    },
                )
            rows.append(row)

    return header, rows


def filter_routes() -> set[str]:
    header, rows = _read_csv(ROUTES_PATH)
    bus_rows: list[list[str]] = []
    mikro_rows: list[list[str]] = []

    for row in rows:
        (mikro_rows if _is_mikrotrans(row[0]) else bus_rows).append(row)

    _write_csv(BUS_ROUTES_PATH, header, bus_rows)
    _write_csv(MIKRO_ROUTES_PATH, header, mikro_rows)

    print(
        "Routes filtered.",
        f"  Bus routes: {len(bus_rows)} -> {BUS_ROUTES_PATH.relative_to(ROOT)}",
        f"  Mikro routes: {len(mikro_rows)} -> {MIKRO_ROUTES_PATH.relative_to(ROOT)}",
    )

    return {row[0] for row in mikro_rows}


def collect_mikrotrans_trip_ids(mikro_route_ids: set[str]) -> set[str]:
    _, trip_rows = _read_csv(TRIPS_PATH)
    mikro_trips = {row[0] for row in trip_rows if row[1] in mikro_route_ids}
    print(f"Identified {len(mikro_trips)} mikrotrans trips from trips.csv")
    return mikro_trips


def filter_stop_times(mikro_trip_ids: set[str]) -> Tuple[set[str], set[str]]:
    header, rows = _read_csv(STOP_TIMES_PATH)
    bus_rows: list[list[str]] = []
    mikro_rows: list[list[str]] = []
    bus_stop_ids: set[str] = set()
    mikro_stop_ids: set[str] = set()

    for row in rows:
        stop_id = row[2]
        if row[0] in mikro_trip_ids:
            mikro_rows.append(row)
            mikro_stop_ids.add(stop_id)
        else:
            bus_rows.append(row)
            bus_stop_ids.add(stop_id)

    _write_csv(STOP_TIMES_PATH, header, bus_rows)
    _write_csv(MIKRO_STOP_TIMES_PATH, header, mikro_rows)

    print(
        "stop_times filtered.",
        f"  Bus stop_times rows: {len(bus_rows)} (written back to {STOP_TIMES_PATH.relative_to(ROOT)})",
        f"  Mikro stop_times rows: {len(mikro_rows)} -> {MIKRO_STOP_TIMES_PATH.relative_to(ROOT)}",
    )

    return bus_stop_ids, mikro_stop_ids


def filter_stops(bus_stop_ids: set[str], mikro_stop_ids: set[str]) -> None:
    header, rows = _read_csv(STOPS_PATH)

    mikro_rows: list[list[str]] = []
    bus_rows: list[list[str]] = []

    mikro_only = mikro_stop_ids - bus_stop_ids

    for row in rows:
        stop_id = row[0]
        if stop_id in mikro_stop_ids:
            mikro_rows.append(row)

        if stop_id not in mikro_only:
            bus_rows.append(row)

    _write_csv(STOPS_PATH, header, bus_rows)
    _write_csv(MIKRO_STOPS_PATH, header, mikro_rows)

    print(
        "stops filtered.",
        f"  Bus stops: {len(bus_rows)} (rewrote {STOPS_PATH.relative_to(ROOT)})",
        f"  Mikro stops: {len(mikro_rows)} -> {MIKRO_STOPS_PATH.relative_to(ROOT)}",
        f"  Mikro-only stops moved out: {len(mikro_only)}",
    )


def _write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    mikro_route_ids = filter_routes()
    mikro_trip_ids = collect_mikrotrans_trip_ids(mikro_route_ids)
    bus_stop_ids, mikro_stop_ids = filter_stop_times(mikro_trip_ids)
    filter_stops(bus_stop_ids, mikro_stop_ids)


if __name__ == "__main__":
    main()
