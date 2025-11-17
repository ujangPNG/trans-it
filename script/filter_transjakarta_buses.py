#!/usr/bin/env python3
"""Filter Transjakarta GTFS so Mikrotrans (JAK.*) data is isolated under filtered/.

Operations performed:
1. `routes.csv` is rewritten to keep only bus routes while Mikrotrans routes are
    stored in `data/transjakarta/filtered/mikrotrans/routes.csv`.
2. `route_list.csv` follows the same rule so the root list is bus-only.
3. `stop_times.csv` and `stops.csv` are rewritten with bus data only, while all
    Mikrotrans rows are exported to `filtered/mikrotrans/`.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
TRANSJAKARTA_DIR = ROOT / "data" / "transjakarta"
ROUTES_PATH = TRANSJAKARTA_DIR / "routes.csv"
ROUTE_LIST_PATH = TRANSJAKARTA_DIR / "route_list.csv"

FILTERED_MIKRO_DIR = TRANSJAKARTA_DIR / "filtered" / "mikrotrans"
MIKRO_ROUTES_PATH = FILTERED_MIKRO_DIR / "routes.csv"
MIKRO_ROUTE_LIST_PATH = FILTERED_MIKRO_DIR / "route_list.csv"

TRIPS_PATH = TRANSJAKARTA_DIR / "trips.csv"
STOP_TIMES_PATH = TRANSJAKARTA_DIR / "stop_times.csv"
STOPS_PATH = TRANSJAKARTA_DIR / "stops.csv"

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


def _read_csv_if_exists(path: Path) -> Tuple[list[str], list[list[str]]] | None:
    if not path.exists():
        return None
    return _read_csv(path)


def filter_routes() -> set[str]:
    header, rows = _read_csv(ROUTES_PATH)
    contains_mikro = any(_is_mikrotrans(row[0]) for row in rows)

    if not contains_mikro:
        extra = _read_csv_if_exists(MIKRO_ROUTES_PATH)
        if extra:
            extra_header, extra_rows = extra
            if extra_header != header:
                raise ValueError("routes.csv header mismatch with mikro routes file")
            rows = rows + extra_rows

    bus_rows: list[list[str]] = []
    mikro_rows: list[list[str]] = []

    for row in rows:
        (mikro_rows if _is_mikrotrans(row[0]) else bus_rows).append(row)

    _write_csv(ROUTES_PATH, header, bus_rows)
    _write_csv(MIKRO_ROUTES_PATH, header, mikro_rows)

    print(
        "Routes filtered.",
        f"  Bus routes: {len(bus_rows)} (rewrote {ROUTES_PATH.relative_to(ROOT)})",
        f"  Mikro routes: {len(mikro_rows)} -> {MIKRO_ROUTES_PATH.relative_to(ROOT)}",
    )

    return {row[0] for row in mikro_rows}


def filter_route_list(mikro_route_ids: set[str]) -> None:
    header, rows = _read_csv(ROUTE_LIST_PATH)
    contains_mikro = any(row[0] in mikro_route_ids for row in rows)

    if not contains_mikro:
        extra = _read_csv_if_exists(MIKRO_ROUTE_LIST_PATH)
        if extra:
            extra_header, extra_rows = extra
            if extra_header != header:
                raise ValueError("route_list.csv header mismatch with mikro route_list file")
            rows = rows + extra_rows

    bus_rows: list[list[str]] = []
    mikro_rows: list[list[str]] = []

    for row in rows:
        if row[0] in mikro_route_ids:
            mikro_rows.append(row)
        else:
            bus_rows.append(row)

    _write_csv(ROUTE_LIST_PATH, header, bus_rows)
    _write_csv(MIKRO_ROUTE_LIST_PATH, header, mikro_rows)

    print(
        "route_list filtered.",
        f"  Bus rows: {len(bus_rows)} (rewrote {ROUTE_LIST_PATH.relative_to(ROOT)})",
        f"  Mikro rows: {len(mikro_rows)} -> {MIKRO_ROUTE_LIST_PATH.relative_to(ROOT)}",
    )


def collect_mikrotrans_trip_ids(mikro_route_ids: set[str]) -> set[str]:
    _, trip_rows = _read_csv(TRIPS_PATH)
    mikro_trips = {row[0] for row in trip_rows if row[1] in mikro_route_ids}
    print(f"Identified {len(mikro_trips)} mikrotrans trips from trips.csv")
    return mikro_trips


def filter_stop_times(mikro_trip_ids: set[str]) -> Tuple[set[str], set[str]]:
    header, rows = _read_csv(STOP_TIMES_PATH)
    contains_mikro = any(row[0] in mikro_trip_ids for row in rows)

    if mikro_trip_ids and not contains_mikro:
        extra = _read_csv_if_exists(MIKRO_STOP_TIMES_PATH)
        if extra:
            extra_header, extra_rows = extra
            if extra_header != header:
                raise ValueError("stop_times.csv header mismatch with mikro stop_times file")
            rows = rows + extra_rows

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
    existing_stop_ids = {row[0] for row in rows}
    missing_mikro_ids = mikro_stop_ids - existing_stop_ids

    if missing_mikro_ids:
        extra = _read_csv_if_exists(MIKRO_STOPS_PATH)
        if extra:
            extra_header, extra_rows = extra
            if extra_header != header:
                raise ValueError("stops.csv header mismatch with mikro stops file")
            for row in extra_rows:
                if row[0] in missing_mikro_ids:
                    rows.append(row)
        else:
            print("Warning: mikrotrans stops missing from both base and filtered datasets")

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
    filter_route_list(mikro_route_ids)
    filter_stops(bus_stop_ids, mikro_stop_ids)


if __name__ == "__main__":
    main()
