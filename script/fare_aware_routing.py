"""Fare-prioritized routing helpers for the TransJakarta network.

The goal is to always minimize the number of paid taps (IDR 3,500 increments)
first, while only using travel time as a tie-breaker when two journeys share
identical fares.

The script can be executed from the CLI to compute the cheapest route between
any two stops. It consumes GTFS files under ``data/transjakarta`` by default
but accepts custom paths, manual integration overrides, and custom transfer
lists via JSON inputs.
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

# Business constants ---------------------------------------------------------
DEFAULT_DATA_DIR = Path("data/transjakarta")
DEFAULT_FARE = 3_500
DEFAULT_TRAVEL_SECONDS = 240  # fallback when GTFS timing data is missing
MIN_TRAVEL_SECONDS = 60  # keeps zero / negative segments out of the graph
DEFAULT_INTERNAL_TRANSFER_SECONDS = 150  # short walk inside the same paid area
DEFAULT_EXTERNAL_TRANSFER_SECONDS = 360  # average walk outside the gate


@dataclass
class PathStep:
    """One leg of the final journey."""

    from_stop_id: str
    to_stop_id: str
    edge_type: str
    cost: int
    time_seconds: float
    attributes: Dict[str, Optional[str]]


@dataclass
class PathResult:
    """Result returned by the fare-aware Dijkstra search."""

    total_fare: int
    total_time_seconds: float
    steps: List[PathStep]


class TransitGraphBuilder:
    """Build a fare-aware directed multigraph from GTFS tables."""

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

        self.stops_df: pd.DataFrame = pd.DataFrame()
        self.trips_df: pd.DataFrame = pd.DataFrame()
        self.stop_times_df: pd.DataFrame = pd.DataFrame()

    def build(self) -> nx.MultiDiGraph:
        """Return a MultiDiGraph with travel/internal/external edges."""

        self._load_frames()
        graph = nx.MultiDiGraph()
        self._add_stop_nodes(graph)
        self._add_travel_edges(graph)
        self._add_internal_transfer_edges(graph)
        self._add_external_transfer_edges(graph)
        return graph

    # Internal helpers ------------------------------------------------------
    def _load_frames(self) -> None:
        self.stops_df = self._read_csv(self.stops_path)
        self.trips_df = self._read_csv(self.trips_path)
        self.stop_times_df = self._read_csv(self.stop_times_path)

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

    def _add_stop_nodes(self, graph: nx.MultiDiGraph) -> None:
        stops = self.stops_df.fillna("")
        for _, row in stops.iterrows():
            stop_id = row.get("stop_id")
            attrs = row.to_dict()
            graph.add_node(stop_id, **attrs)

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
                        cost=0,  # Riding the bus does not incur extra fare.
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
            # Handle trips that cross midnight by rolling the arrival forward.
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
            # External transfers require exiting the paid area, hence the penalty.
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
    """Return seconds since midnight for a GTFS HH:MM:SS value."""

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
    """Run Dijkstra with (fare, time) lexicographic ordering."""

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


def _is_improved(
    new_fare: int,
    new_time: float,
    existing: Tuple[int, float],
) -> bool:
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


def resolve_stop_id(graph: nx.MultiDiGraph, query: str) -> str:
    """Allow specifying either the stop_id or an exact/partial stop name."""

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


def format_path_summary(
    result: PathResult,
    graph: nx.MultiDiGraph,
) -> str:
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
        lines.append(
            f"  {idx:02d}. {origin_name} -> {dest_name} | {descriptor}{extra_str} | "
            f"cost +{step.cost} | {step.time_seconds / 60:.1f} min",
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute the cheapest TransJakarta route by minimizing fare first "
            "and time second."
        )
    )
    parser.add_argument("--stops-path", type=Path, default=DEFAULT_DATA_DIR / "stops.csv")
    parser.add_argument("--trips-path", type=Path, default=DEFAULT_DATA_DIR / "trips.csv")
    parser.add_argument(
        "--stop-times-path",
        type=Path,
        default=DEFAULT_DATA_DIR / "stop_times.csv",
    )
    parser.add_argument("--origin", required=True, help="Stop ID or exact/partial name")
    parser.add_argument("--destination", required=True, help="Stop ID or exact/partial name")
    parser.add_argument(
        "--fare-penalty",
        type=int,
        default=DEFAULT_FARE,
        help="Cost of exiting and tapping back in (IDR)",
    )
    parser.add_argument(
        "--internal-transfer-seconds",
        type=int,
        default=DEFAULT_INTERNAL_TRANSFER_SECONDS,
        help="Assumed walking time for transfers inside the same paid area",
    )
    parser.add_argument(
        "--external-transfer-seconds",
        type=int,
        default=DEFAULT_EXTERNAL_TRANSFER_SECONDS,
        help="Fallback walking time when leaving the paid area",
    )
    parser.add_argument(
        "--manual-internal-groups",
        type=str,
        help="Path to JSON defining additional integrated halte groups",
    )
    parser.add_argument(
        "--external-transfer-config",
        type=str,
        help="Path to JSON describing walk transfers that require a new fare",
    )
    parser.add_argument(
        "--default-travel-seconds",
        type=int,
        default=DEFAULT_TRAVEL_SECONDS,
        help="Fallback travel time for segments missing schedule data",
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
    )
    graph = builder.build()
    source = resolve_stop_id(graph, args.origin)
    target = resolve_stop_id(graph, args.destination)
    result = fare_priority_dijkstra(graph, source, target)
    print(format_path_summary(result, graph))


if __name__ == "__main__":
    main()
