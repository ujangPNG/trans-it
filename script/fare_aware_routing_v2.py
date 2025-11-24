"""Second iteration of the fare-aware TransJakarta router.

Differences vs v1:
- Can enrich GTFS stops with coordinates + metadata from `data-lokasi-halte.csv`.
- Accepts lat/lon inputs to auto-select the closest origin/destination halte.
- Formats output with coordinates so users can validate the chosen nodes.

The original `fare_aware_routing.py` remains untouched for backwards compatibility.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx
import pandas as pd

DEFAULT_DATA_DIR = Path("data/transjakarta")
DEFAULT_FARE = 3_500
DEFAULT_TRAVEL_SECONDS = 240
MIN_TRAVEL_SECONDS = 60
DEFAULT_INTERNAL_TRANSFER_SECONDS = 150
DEFAULT_EXTERNAL_TRANSFER_SECONDS = 360
DEFAULT_HALTE_LOCATIONS_PATH = DEFAULT_DATA_DIR / "data-lokasi-halte.csv"
EARTH_RADIUS_METERS = 6_371_000


@dataclass
class PathStep:
    from_stop_id: str
    to_stop_id: str
    edge_type: str
    cost: int
    time_seconds: float
    attributes: Dict[str, Optional[str]]


@dataclass
class PathResult:
    total_fare: int
    total_time_seconds: float
    steps: List[PathStep]


class TransitGraphBuilder:
    def __init__(
        self,
        stops_path: Path,
        trips_path: Path,
        stop_times_path: Path,
        *,
        fare_penalty: int = DEFAULT_FARE,
        internal_transfer_seconds: int = DEFAULT_INTERNAL_TRANSFER_SECONDS,
        external_transfer_seconds: int = DEFAULT_EXTERNAL_TRANSFER_SECONDS,
        default_travel_seconds: int = DEFAULT_TRAVEL_SECONDS,
        manual_internal_groups: Optional[Dict[str, List[str]]] = None,
        external_transfers: Optional[List[Dict[str, object]]] = None,
        halte_locations_path: Optional[Path] = DEFAULT_HALTE_LOCATIONS_PATH,
    ) -> None:
        self.stops_path = stops_path
        self.trips_path = trips_path
        self.stop_times_path = stop_times_path
        self.fare_penalty = fare_penalty
        self.internal_transfer_seconds = internal_transfer_seconds
        self.external_transfer_seconds = external_transfer_seconds
        self.default_travel_seconds = default_travel_seconds
        self.manual_internal_groups = manual_internal_groups or {}
        self.external_transfers = external_transfers or []
        self.halte_locations_path = halte_locations_path

        self.stops_df: pd.DataFrame = pd.DataFrame()
        self.trips_df: pd.DataFrame = pd.DataFrame()
        self.stop_times_df: pd.DataFrame = pd.DataFrame()
        self.halte_locations_df: pd.DataFrame = pd.DataFrame()
        self.location_lookup: Dict[str, List[Dict[str, object]]] = {}

    def build(self) -> nx.MultiDiGraph:
        self._load_frames()
        graph = nx.MultiDiGraph()
        self._add_stop_nodes(graph)
        self._add_travel_edges(graph)
        self._add_internal_transfer_edges(graph)
        self._add_external_transfer_edges(graph)
        return graph

    def _load_frames(self) -> None:
        self.stops_df = self._read_csv(self.stops_path)
        self.trips_df = self._read_csv(self.trips_path)
        self.stop_times_df = self._read_csv(self.stop_times_path)
        self._load_halte_locations()

        for frame, column in (
            (self.stops_df, "stop_id"),
            (self.trips_df, "trip_id"),
            (self.stop_times_df, "stop_id"),
        ):
            if column not in frame.columns:
                available = ", ".join(frame.columns)
                raise KeyError(
                    f"Column '{column}' is missing from {frame.__class__.__name__}; "
                    f"available columns: {available}"
                )
            frame[column] = frame[column].astype(str)

    def _read_csv(self, path: Path) -> pd.DataFrame:
        frame = pd.read_csv(path, skipinitialspace=True)
        frame.columns = [str(col).strip() for col in frame.columns]
        return frame

    def _load_halte_locations(self) -> None:
        if not self.halte_locations_path or not self.halte_locations_path.exists():
            self.halte_locations_df = pd.DataFrame()
            self.location_lookup = {}
            return
        frame = self._read_csv(self.halte_locations_path)
        rename_map = {
            "wilayah": "region",
            "lokasi_alamat": "address",
            "lintang": "latitude",
            "bujur": "longitude",
            "no_registrasi": "location_id",
        }
        frame = frame.rename(columns=rename_map)
        frame["address"] = frame["address"].fillna("")
        frame["halte_name"] = frame["address"].apply(extract_halte_name)
        frame["lookup_key"] = frame["halte_name"].apply(normalize_stop_name)
        frame = frame.loc[frame["lookup_key"].str.len() > 0]
        self.halte_locations_df = frame
        lookup: Dict[str, List[Dict[str, object]]] = {}
        for _, row in frame.iterrows():
            lookup.setdefault(row["lookup_key"], []).append(row.to_dict())
        self.location_lookup = lookup

    def _add_stop_nodes(self, graph: nx.MultiDiGraph) -> None:
        stops = self.stops_df.fillna("")
        for _, row in stops.iterrows():
            stop_id = row.get("stop_id")
            attrs = row.to_dict()
            enriched = self._enrich_with_location_data(attrs)
            graph.add_node(stop_id, **enriched)
        self._index_coordinates(graph)

    def _enrich_with_location_data(self, attrs: Dict[str, object]) -> Dict[str, object]:
        stop_name = str(attrs.get("stop_name", ""))
        lookup_key = normalize_stop_name(stop_name)
        candidates = self.location_lookup.get(lookup_key, [])
        stop_lat = to_optional_float(attrs.get("stop_lat"))
        stop_lon = to_optional_float(attrs.get("stop_lon"))
        best_match: Optional[Dict[str, object]] = None
        if candidates:
            if stop_lat is not None and stop_lon is not None:
                best_match = min(
                    candidates,
                    key=lambda item: geo_distance_meters(
                        stop_lat,
                        stop_lon,
                        to_optional_float(item.get("latitude")) or stop_lat,
                        to_optional_float(item.get("longitude")) or stop_lon,
                    ),
                )
            else:
                best_match = candidates[0]

        if best_match:
            ext_lat = to_optional_float(best_match.get("latitude"))
            ext_lon = to_optional_float(best_match.get("longitude"))
            if stop_lat is None and ext_lat is not None:
                attrs["stop_lat"] = ext_lat
            if stop_lon is None and ext_lon is not None:
                attrs["stop_lon"] = ext_lon
            attrs.setdefault("location_address", best_match.get("address"))
            attrs.setdefault("location_region", best_match.get("region"))
            attrs.setdefault("location_id", best_match.get("location_id"))
            attrs.setdefault("location_source", "data-lokasi-halte.csv")
            attrs.setdefault("location_latitude", ext_lat)
            attrs.setdefault("location_longitude", ext_lon)
        return attrs

    def _index_coordinates(self, graph: nx.MultiDiGraph) -> None:
        for node_id, data in graph.nodes(data=True):
            lat = to_optional_float(data.get("stop_lat"))
            lon = to_optional_float(data.get("stop_lon"))
            if lat is None or lon is None:
                continue
            graph.nodes[node_id]["stop_lat"] = lat
            graph.nodes[node_id]["stop_lon"] = lon

    def _add_travel_edges(self, graph: nx.MultiDiGraph) -> None:
        merged = self.stop_times_df.merge(
            self.trips_df[["trip_id", "route_id", "service_id"]],
            on="trip_id",
            how="left",
        )
        merged = merged.sort_values(["trip_id", "stop_sequence"])

        for trip_id, group in merged.groupby("trip_id", sort=False):
            prev_row = None
            for _, row in group.iterrows():
                if prev_row is not None:
                    origin = prev_row["stop_id"]
                    dest = row["stop_id"]
                    if origin == dest:
                        prev_row = row
                        continue
                    travel_seconds = self._segment_duration_seconds(prev_row, row)
                    graph.add_edge(
                        origin,
                        dest,
                        edge_type="travel",
                        cost=0,
                        time=travel_seconds,
                        route_id=row.get("route_id"),
                        trip_id=trip_id,
                        service_id=row.get("service_id"),
                        departure_time=prev_row.get("departure_time"),
                        arrival_time=row.get("arrival_time"),
                    )
                prev_row = row

    def _segment_duration_seconds(self, row_a: pd.Series, row_b: pd.Series) -> float:
        depart = parse_gtfs_time(row_a.get("departure_time") or row_a.get("arrival_time"))
        arrive = parse_gtfs_time(row_b.get("arrival_time") or row_b.get("departure_time"))
        if depart is not None and arrive is not None:
            if arrive < depart:
                arrive += 24 * 3600
            delta = max(arrive - depart, MIN_TRAVEL_SECONDS)
            return float(delta)
        return float(self.default_travel_seconds)

    def _add_internal_transfer_edges(self, graph: nx.MultiDiGraph) -> None:
        parent_groups = (
            self.stops_df.fillna("")
            .loc[lambda df: df["parent_station"].astype(str).str.len() > 0]
            .groupby("parent_station")
        )
        for parent_id, group in parent_groups:
            self._connect_group_nodes(
                graph,
                group["stop_id"].tolist(),
                edge_type="internal-transfer",
                seconds=self.internal_transfer_seconds,
                attributes={"parent_station": parent_id},
            )

        for name, stop_ids in self.manual_internal_groups.items():
            self._connect_group_nodes(
                graph,
                stop_ids,
                edge_type="internal-transfer",
                seconds=self.internal_transfer_seconds,
                attributes={"manual_group": name},
            )

    def _add_external_transfer_edges(self, graph: nx.MultiDiGraph) -> None:
        for item in self.external_transfers:
            from_stop = str(item.get("from_stop_id"))
            to_stop = str(item.get("to_stop_id"))
            if from_stop not in graph or to_stop not in graph:
                continue
            seconds = float(item.get("time_seconds", self.external_transfer_seconds))
            fare = int(item.get("fare", self.fare_penalty))
            bidirectional = bool(item.get("bidirectional", True))
            attrs = {"name": item.get("name"), "notes": item.get("notes")}
            graph.add_edge(
                from_stop,
                to_stop,
                edge_type="external-transfer",
                cost=fare,
                time=seconds,
                **attrs,
            )
            if bidirectional:
                graph.add_edge(
                    to_stop,
                    from_stop,
                    edge_type="external-transfer",
                    cost=fare,
                    time=seconds,
                    **attrs,
                )

    def _connect_group_nodes(
        self,
        graph: nx.MultiDiGraph,
        stop_ids: Iterable[str],
        *,
        edge_type: str,
        seconds: int,
        attributes: Optional[Dict[str, str]] = None,
    ) -> None:
        nodes = [stop_id for stop_id in stop_ids if stop_id in graph]
        if len(nodes) < 2:
            return
        attrs = attributes or {}
        for a, b in combinations(nodes, 2):
            graph.add_edge(
                a,
                b,
                edge_type=edge_type,
                cost=0,
                time=float(seconds),
                **attrs,
            )
            graph.add_edge(
                b,
                a,
                edge_type=edge_type,
                cost=0,
                time=float(seconds),
                **attrs,
            )


def parse_gtfs_time(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    value_str = str(value)
    if not value_str or value_str.lower() == "nan":
        return None
    parts = value_str.split(":")
    if len(parts) != 3:
        return None
    try:
        hours, minutes, seconds = (int(part) for part in parts)
    except ValueError:
        return None
    return hours * 3600 + minutes * 60 + seconds


def fare_priority_dijkstra(
    graph: nx.MultiDiGraph,
    source: str,
    target: str,
) -> PathResult:
    if source not in graph:
        raise KeyError(f"Unknown source stop_id '{source}'")
    if target not in graph:
        raise KeyError(f"Unknown target stop_id '{target}'")

    heap: List[Tuple[int, float, str]] = [(0, 0.0, source)]
    best: Dict[str, Tuple[int, float]] = {source: (0, 0.0)}
    parent: Dict[str, Tuple[str, Dict[str, object]]] = {}

    while heap:
        fare, elapsed, node = heapq.heappop(heap)
        if node == target:
            steps = _reconstruct_path(parent, source, target)
            return PathResult(fare, elapsed, steps)
        best_fare, best_time = best[node]
        if fare > best_fare or (fare == best_fare and elapsed > best_time):
            continue
        for neighbor in graph.successors(node):
            edge_dict = graph.get_edge_data(node, neighbor)
            if not edge_dict:
                continue
            for edge_data in edge_dict.values():
                edge_cost = int(edge_data.get("cost", 0))
                edge_time = float(edge_data.get("time", DEFAULT_TRAVEL_SECONDS))
                new_fare = fare + edge_cost
                new_time = elapsed + edge_time
                best_neighbor = best.get(neighbor)
                if best_neighbor is None or _is_improved(new_fare, new_time, best_neighbor):
                    best[neighbor] = (new_fare, new_time)
                    parent[neighbor] = (node, edge_data)
                    heapq.heappush(heap, (new_fare, new_time, neighbor))

    raise ValueError(f"No path between {source} and {target} was found")


def _is_improved(new_fare: int, new_time: float, existing: Tuple[int, float]) -> bool:
    fare, time = existing
    if new_fare < fare:
        return True
    if new_fare == fare and new_time < time:
        return True
    return False


def _reconstruct_path(
    parent: Dict[str, Tuple[str, Dict[str, object]]],
    source: str,
    target: str,
) -> List[PathStep]:
    steps: List[PathStep] = []
    node = target
    while node != source:
        if node not in parent:
            break
        prev_node, edge_data = parent[node]
        attributes = {
            key: value
            for key, value in edge_data.items()
            if key not in {"cost", "time"}
        }
        steps.append(
            PathStep(
                from_stop_id=prev_node,
                to_stop_id=node,
                edge_type=str(edge_data.get("edge_type", "travel")),
                cost=int(edge_data.get("cost", 0)),
                time_seconds=float(edge_data.get("time", DEFAULT_TRAVEL_SECONDS)),
                attributes=attributes,
            )
        )
        node = prev_node
    steps.reverse()
    return steps


def resolve_stop_id(
    graph: nx.MultiDiGraph,
    query: Optional[str],
    *,
    coordinate: Optional[Tuple[float, float]] = None,
) -> str:
    if coordinate is not None:
        return nearest_stop_by_coordinate(graph, coordinate)
    if not query:
        raise ValueError("Stop query is empty and no coordinates were provided")
    query = query.strip()
    if query in graph:
        return query
    query_lower = query.lower()
    matches = [
        node
        for node, data in graph.nodes(data=True)
        if str(data.get("stop_name", "")).lower() == query_lower
    ]
    if not matches:
        matches = [
            node
            for node, data in graph.nodes(data=True)
            if query_lower in str(data.get("stop_name", "")).lower()
        ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous stop '{query}'. Candidates: {matches}")
    raise KeyError(f"Stop '{query}' was not found by id or name")


def nearest_stop_by_coordinate(
    graph: nx.MultiDiGraph,
    coordinate: Tuple[float, float],
) -> str:
    lat, lon = coordinate
    best_node: Optional[str] = None
    best_distance = float("inf")
    for node, data in graph.nodes(data=True):
        stop_lat = to_optional_float(data.get("stop_lat"))
        stop_lon = to_optional_float(data.get("stop_lon"))
        if stop_lat is None or stop_lon is None:
            continue
        distance = geo_distance_meters(lat, lon, stop_lat, stop_lon)
        if distance < best_distance:
            best_distance = distance
            best_node = node
    if best_node is None:
        raise ValueError("No stops with coordinate data are available")
    return best_node


def load_manual_groups(path: Optional[str]) -> Dict[str, List[str]]:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, dict):
        return {key: list(value) for key, value in payload.items()}
    if isinstance(payload, list):
        result: Dict[str, List[str]] = {}
        for idx, entry in enumerate(payload):
            name = str(entry.get("id") or entry.get("name") or f"group_{idx}")
            stop_ids = entry.get("stop_ids")
            if not stop_ids:
                continue
            result[name] = [str(stop_id) for stop_id in stop_ids]
        return result
    raise ValueError("Manual group config must be a dict or a list")


def load_external_transfers(path: Optional[str]) -> List[Dict[str, object]]:
    if not path:
        return []
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, dict):
        payload = payload.get("transfers")
    if not isinstance(payload, list):
        raise ValueError("External transfer config must be a list or contain one")
    return payload


def format_path_summary(result: PathResult, graph: nx.MultiDiGraph) -> str:
    lines = [
        f"Cheapest fare: IDR {result.total_fare:,}",
        f"Estimated time: {result.total_time_seconds / 60:.1f} minutes",
        "Steps:",
    ]
    for idx, step in enumerate(result.steps, start=1):
        origin_name = graph.nodes[step.from_stop_id].get("stop_name", step.from_stop_id)
        dest_name = graph.nodes[step.to_stop_id].get("stop_name", step.to_stop_id)
        descriptor = step.edge_type.replace("-", " ")
        extra = step.attributes.get("route_id") or step.attributes.get("name")
        extra_str = f" ({extra})" if extra else ""
        dest_lat = graph.nodes[step.to_stop_id].get("stop_lat")
        dest_lon = graph.nodes[step.to_stop_id].get("stop_lon")
        coord_str = (
            f" | ({dest_lat:.5f}, {dest_lon:.5f})"
            if isinstance(dest_lat, (int, float)) and isinstance(dest_lon, (int, float))
            else ""
        )
        lines.append(
            f"  {idx:02d}. {origin_name} -> {dest_name} | {descriptor}{extra_str} | "
            f"cost +{step.cost} | {step.time_seconds / 60:.1f} min{coord_str}",
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cheapest TransJakarta route with coordinate-aware origin/destination support",
    )
    parser.add_argument("--stops-path", type=Path, default=DEFAULT_DATA_DIR / "stops.csv")
    parser.add_argument("--trips-path", type=Path, default=DEFAULT_DATA_DIR / "trips.csv")
    parser.add_argument(
        "--stop-times-path",
        type=Path,
        default=DEFAULT_DATA_DIR / "stop_times.csv",
    )
    parser.add_argument("--origin", help="Stop ID or exact/partial name")
    parser.add_argument("--destination", help="Stop ID or exact/partial name")
    parser.add_argument("--origin-lat", type=float, help="Latitude to locate the nearest origin halte")
    parser.add_argument("--origin-lon", type=float, help="Longitude to locate the nearest origin halte")
    parser.add_argument(
        "--destination-lat",
        type=float,
        help="Latitude to locate the nearest destination halte",
    )
    parser.add_argument(
        "--destination-lon",
        type=float,
        help="Longitude to locate the nearest destination halte",
    )
    parser.add_argument("--fare-penalty", type=int, default=DEFAULT_FARE)
    parser.add_argument(
        "--internal-transfer-seconds",
        type=int,
        default=DEFAULT_INTERNAL_TRANSFER_SECONDS,
    )
    parser.add_argument(
        "--external-transfer-seconds",
        type=int,
        default=DEFAULT_EXTERNAL_TRANSFER_SECONDS,
    )
    parser.add_argument("--manual-internal-groups", type=str)
    parser.add_argument("--external-transfer-config", type=str)
    parser.add_argument("--default-travel-seconds", type=int, default=DEFAULT_TRAVEL_SECONDS)
    parser.add_argument(
        "--halte-locations-path",
        type=Path,
        default=DEFAULT_HALTE_LOCATIONS_PATH,
        help="CSV containing halte metadata (data-lokasi-halte.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manual_groups = load_manual_groups(args.manual_internal_groups)
    external_transfers = load_external_transfers(args.external_transfer_config)
    builder = TransitGraphBuilder(
        stops_path=args.stops_path,
        trips_path=args.trips_path,
        stop_times_path=args.stop_times_path,
        fare_penalty=args.fare_penalty,
        internal_transfer_seconds=args.internal_transfer_seconds,
        external_transfer_seconds=args.external_transfer_seconds,
        default_travel_seconds=args.default_travel_seconds,
        manual_internal_groups=manual_groups,
        external_transfers=external_transfers,
        halte_locations_path=args.halte_locations_path,
    )
    graph = builder.build()
    _validate_coordinate_args(args)
    source_coord = build_coordinate_tuple(args.origin_lat, args.origin_lon)
    dest_coord = build_coordinate_tuple(args.destination_lat, args.destination_lon)
    source = resolve_stop_id(graph, args.origin, coordinate=source_coord)
    target = resolve_stop_id(graph, args.destination, coordinate=dest_coord)
    result = fare_priority_dijkstra(graph, source, target)
    print(format_path_summary(result, graph))


# Helper utilities ---------------------------------------------------------

def normalize_stop_name(value: str) -> str:
    return " ".join(value.lower().strip().split())


def extract_halte_name(address: str) -> str:
    if "(" in address and ")" in address:
        inside = address[address.rfind("(") + 1 : address.rfind(")")]
        return inside.replace("Halte", "").strip() or address
    return address.replace("Halte", "").strip()


def to_optional_float(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def geo_distance_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_METERS * c


def build_coordinate_tuple(lat: Optional[float], lon: Optional[float]) -> Optional[Tuple[float, float]]:
    if lat is None and lon is None:
        return None
    if lat is None or lon is None:
        raise ValueError("Both latitude and longitude must be provided when using coordinates")
    return (lat, lon)


def _validate_coordinate_args(args: argparse.Namespace) -> None:
    if (args.origin_lat is None) != (args.origin_lon is None):
        raise ValueError("Origin latitude and longitude must be provided together")
    if (args.destination_lat is None) != (args.destination_lon is None):
        raise ValueError("Destination latitude and longitude must be provided together")


if __name__ == "__main__":
    main()
