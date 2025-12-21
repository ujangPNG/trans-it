"""FastAPI backend that exposes the fare-aware routing engine on port 25200."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel, ConfigDict, Field
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from script.fare_aware_routing_v2 import (  # noqa: E402
    DEFAULT_DATA_DIR,
    DEFAULT_EXTERNAL_TRANSFER_SECONDS,
    DEFAULT_FARE,
    DEFAULT_HALTE_LOCATIONS_PATH,
    DEFAULT_INTERNAL_TRANSFER_SECONDS,
    DEFAULT_TRAVEL_SECONDS,
    PathStep,
    TransitGraphBuilder,
    build_coordinate_tuple,
    fare_priority_dijkstra,
    format_path_summary,
    load_external_transfers,
    load_manual_groups,
    resolve_stop_id,
    to_optional_float,
)


@dataclass
class RouterSettings:
    stops_path: Path = DEFAULT_DATA_DIR / "stops.csv"
    trips_path: Path = DEFAULT_DATA_DIR / "trips.csv"
    stop_times_path: Path = DEFAULT_DATA_DIR / "stop_times.csv"
    halte_locations_path: Path = DEFAULT_HALTE_LOCATIONS_PATH
    fare_penalty: int = DEFAULT_FARE
    internal_transfer_seconds: int = DEFAULT_INTERNAL_TRANSFER_SECONDS
    external_transfer_seconds: int = DEFAULT_EXTERNAL_TRANSFER_SECONDS
    default_travel_seconds: int = DEFAULT_TRAVEL_SECONDS
    manual_internal_groups_path: Optional[Path] = None
    external_transfer_config_path: Optional[Path] = None


def _path_from_env(var_name: str, default: Path) -> Path:
    value = os.getenv(var_name)
    if value:
        return Path(value)
    return default


def _optional_path_from_env(var_name: str) -> Optional[Path]:
    value = os.getenv(var_name)
    if value:
        return Path(value)
    return None


SETTINGS = RouterSettings(
    stops_path=_path_from_env("TRANSIT_STOPS_PATH", DEFAULT_DATA_DIR / "stops.csv"),
    trips_path=_path_from_env("TRANSIT_TRIPS_PATH", DEFAULT_DATA_DIR / "trips.csv"),
    stop_times_path=_path_from_env(
        "TRANSIT_STOP_TIMES_PATH", DEFAULT_DATA_DIR / "stop_times.csv"
    ),
    halte_locations_path=_path_from_env(
        "TRANSIT_HALTE_LOCATIONS_PATH", DEFAULT_HALTE_LOCATIONS_PATH
    ),
    fare_penalty=int(os.getenv("TRANSIT_FARE_PENALTY", DEFAULT_FARE)),
    internal_transfer_seconds=int(
        os.getenv("TRANSIT_INTERNAL_TRANSFER_SECONDS", DEFAULT_INTERNAL_TRANSFER_SECONDS)
    ),
    external_transfer_seconds=int(
        os.getenv("TRANSIT_EXTERNAL_TRANSFER_SECONDS", DEFAULT_EXTERNAL_TRANSFER_SECONDS)
    ),
    default_travel_seconds=int(
        os.getenv("TRANSIT_DEFAULT_TRAVEL_SECONDS", DEFAULT_TRAVEL_SECONDS)
    ),
    manual_internal_groups_path=_optional_path_from_env("TRANSIT_MANUAL_GROUPS_PATH"),
    external_transfer_config_path=_optional_path_from_env("TRANSIT_EXTERNAL_TRANSFERS_PATH"),
)


class Coordinate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    lat: float = Field(..., alias="latitude")
    lon: float = Field(..., alias="longitude")


class StepResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    from_stop_id: str = Field(..., alias="fromStopId")
    from_stop_name: Optional[str] = Field(None, alias="fromStopName")
    to_stop_id: str = Field(..., alias="toStopId")
    to_stop_name: Optional[str] = Field(None, alias="toStopName")
    edge_type: str = Field(..., alias="edgeType")
    cost: int
    time_seconds: float = Field(..., alias="timeSeconds")
    route_id: Optional[str] = Field(None, alias="routeId")
    notes: Optional[str] = None
    to_coordinates: Optional[Coordinate] = Field(None, alias="toCoordinates")
    available_routes: List[str] = Field(
        default_factory=list,
        alias="availableRoutes",
        description="All bus routes available at the relevant stop",
    )
    recommended_routes: List[str] = Field(
        default_factory=list,
        alias="recommendedRoutes",
        description="Routes that directly serve the next stop in this step",
    )
    is_transfer: bool = Field(False, alias="isTransfer", description="True when this step is a transfer between routes")


class RouteResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    total_fare: int = Field(..., alias="totalFare")
    total_time_seconds: float = Field(..., alias="totalTimeSeconds")
    steps: List[StepResponse]
    summary: str


class RouteRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    origin: Optional[str] = Field(
        None, description="Stop ID or name for the origin halte"
    )
    destination: Optional[str] = Field(
        None, description="Stop ID or name for the destination halte"
    )
    origin_lat: Optional[float] = Field(
        None, alias="originLat", description="Latitude for origin autoselection"
    )
    origin_lon: Optional[float] = Field(
        None, alias="originLon", description="Longitude for origin autoselection"
    )
    destination_lat: Optional[float] = Field(
        None, alias="destinationLat", description="Latitude for destination autoselection"
    )
    destination_lon: Optional[float] = Field(
        None, alias="destinationLon", description="Longitude for destination autoselection"
    )


class RoutingEngine:
    def __init__(self, settings: RouterSettings) -> None:
        self.settings = settings
        self.graph = self._build_graph()
        self._routes_by_stop: Dict[str, List[str]] = {}
        self._build_routes_cache()

    def _build_graph(self) -> nx.MultiDiGraph:
        manual_groups: Dict[str, List[str]] = {}
        if self.settings.manual_internal_groups_path and self.settings.manual_internal_groups_path.exists():
            manual_groups = load_manual_groups(str(self.settings.manual_internal_groups_path))

        external_transfers: List[Dict[str, object]] = []
        if self.settings.external_transfer_config_path and self.settings.external_transfer_config_path.exists():
            external_transfers = load_external_transfers(
                str(self.settings.external_transfer_config_path)
            )

        builder = TransitGraphBuilder(
            stops_path=self.settings.stops_path,
            trips_path=self.settings.trips_path,
            stop_times_path=self.settings.stop_times_path,
            fare_penalty=self.settings.fare_penalty,
            internal_transfer_seconds=self.settings.internal_transfer_seconds,
            external_transfer_seconds=self.settings.external_transfer_seconds,
            default_travel_seconds=self.settings.default_travel_seconds,
            manual_internal_groups=manual_groups,
            external_transfers=external_transfers,
            halte_locations_path=self.settings.halte_locations_path,
        )
        return builder.build()

    def _build_routes_cache(self) -> None:
        """Build a cache of all routes that pass through each stop and stop-to-stop connections."""
        import csv
        from collections import defaultdict
        
        routes_by_stop: Dict[str, set[str]] = defaultdict(set)
        # Map from (from_stop, to_stop) to set of route_ids that connect them
        self._routes_between_stops: Dict[Tuple[str, str], set[str]] = defaultdict(set)
        
        # Read trips to get route_id for each trip_id
        trip_to_route: Dict[str, str] = {}
        if self.settings.trips_path.exists():
            with self.settings.trips_path.open('r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    trip_id = row.get('trip_id', '').strip()
                    route_id = row.get('route_id', '').strip()
                    if trip_id and route_id:
                        trip_to_route[trip_id] = route_id
        
        # Read stop_times to map stops to routes and build connections
        if self.settings.stop_times_path.exists():
            # Group by trip_id to track sequences
            trip_stops: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
            
            with self.settings.stop_times_path.open('r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    trip_id = row.get('trip_id', '').strip()
                    stop_id = row.get('stop_id', '').strip()
                    stop_sequence = row.get('stop_sequence', '').strip()
                    
                    if trip_id in trip_to_route and stop_id:
                        route_id = trip_to_route[trip_id]
                        routes_by_stop[stop_id].add(route_id)
                        
                        # Track stop sequence for this trip
                        try:
                            seq = int(stop_sequence) if stop_sequence else 0
                            trip_stops[trip_id].append((seq, stop_id))
                        except ValueError:
                            pass
            
            # Build stop-to-stop connections
            for trip_id, stops_list in trip_stops.items():
                if trip_id not in trip_to_route:
                    continue
                    
                route_id = trip_to_route[trip_id]
                # Sort by sequence
                stops_list.sort(key=lambda x: x[0])
                
                # Create edges for consecutive stops
                for i in range(len(stops_list) - 1):
                    from_stop = stops_list[i][1]
                    to_stop = stops_list[i + 1][1]
                    self._routes_between_stops[(from_stop, to_stop)].add(route_id)
        
        # Convert sets to sorted lists
        self._routes_by_stop = {
            stop_id: sorted(routes) 
            for stop_id, routes in routes_by_stop.items()
        }

    def get_routes_at_stop(self, stop_id: str, next_stop_id: Optional[str] = None) -> List[str]:
        """Get bus routes at the given stop.
        
        If next_stop_id is provided, only return routes that go from stop_id to next_stop_id.
        Otherwise, return all routes that pass through stop_id.
        """
        if next_stop_id:
            # Return only routes that connect these two stops
            routes = self._routes_between_stops.get((stop_id, next_stop_id), set())
            return sorted(routes)
        else:
            # Return all routes at this stop
            return self._routes_by_stop.get(stop_id, [])

    def rebuild(self) -> None:
        self.graph = self._build_graph()
        self._build_routes_cache()


@lru_cache(maxsize=1)
def get_engine() -> RoutingEngine:
    return RoutingEngine(SETTINGS)


app = FastAPI(
    title="TransJakarta Fare-Aware Routing API",
    version="0.2.0",
    description="Backend facade over the fare-aware routing engine (port 25200).",
)

INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY")


def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> str:
    """Verify that the API key matches the configured INTERNAL_API_KEY."""
    if not INTERNAL_API_KEY:
        raise HTTPException(status_code=500, detail="API key not configured")
    if x_api_key != INTERNAL_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return x_api_key


@app.get("/health")
def healthcheck() -> Dict[str, object]:
    engine = get_engine()
    return {
        "status": "ok",
        "stops": engine.graph.number_of_nodes(),
        "edges": engine.graph.number_of_edges(),
    }

@app.get("/")
def tes():
    """tester"""
    return {
        "tes":"ya"
    }

@app.post("/route", response_model=RouteResponse)
def compute_route(
    payload: RouteRequest, api_key: str = Depends(verify_api_key)
) -> RouteResponse:
    engine = get_engine()
    if payload.origin is None and payload.origin_lat is None and payload.origin_lon is None:
        raise HTTPException(status_code=400, detail="origin or originLat/originLon is required")
    if (
        payload.destination is None
        and payload.destination_lat is None
        and payload.destination_lon is None
    ):
        raise HTTPException(
            status_code=400, detail="destination or destinationLat/destinationLon is required"
        )

    source_coord = _build_optional_coordinate(payload.origin_lat, payload.origin_lon)
    dest_coord = _build_optional_coordinate(payload.destination_lat, payload.destination_lon)

    try:
        source_id = resolve_stop_id(engine.graph, payload.origin or "", coordinate=source_coord)
        target_id = resolve_stop_id(
            engine.graph, payload.destination or "", coordinate=dest_coord
        )
    except (KeyError, ValueError) as err:
        raise HTTPException(status_code=404, detail=str(err)) from err

    try:
        result = fare_priority_dijkstra(engine.graph, source_id, target_id)
    except ValueError as err:
        raise HTTPException(status_code=404, detail=str(err)) from err

    summary = format_path_summary(result, engine.graph)
    steps = [_step_to_response(step, engine.graph, engine) for step in result.steps]
    return RouteResponse(
        total_fare=result.total_fare,
        total_time_seconds=result.total_time_seconds,
        steps=steps,
        summary=summary,
    )


def _build_optional_coordinate(
    lat: Optional[float], lon: Optional[float]
) -> Optional[tuple[float, float]]:
    if lat is None and lon is None:
        return None
    try:
        return build_coordinate_tuple(lat, lon)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err


def _step_to_response(step: PathStep, graph: nx.MultiDiGraph, engine: RoutingEngine) -> StepResponse:
    # Translate an internal PathStep into the API-friendly StepResponse.
    from_attrs = graph.nodes.get(step.from_stop_id, {})
    to_attrs = graph.nodes.get(step.to_stop_id, {})
    to_coordinates = _coords_from_attrs(to_attrs)

    # Get route_id, handling potential nan values
    route_id = step.attributes.get("route_id") or step.attributes.get("name")
    if route_id is not None:
        try:
            # Convert to string and check if it's valid
            route_id_str = str(route_id)
            if route_id_str.lower() in ('nan', 'none', ''):
                route_id = None
            else:
                route_id = route_id_str
        except Exception:
            route_id = None

    is_transfer = step.edge_type != "travel"

    # For travel edges: recommended_routes are the routes that directly connect from_stop -> to_stop.
    # available_routes are all routes at the origin stop (what the rider can board here).
    # If file-derived lookups are empty, fall back to graph edges so the user still gets guidance.
    # Fallback order for route_id: edge attributes -> recommended_routes -> available_routes.
    if not is_transfer:
        recommended_routes = engine.get_routes_at_stop(step.from_stop_id, step.to_stop_id)
        available_routes = engine.get_routes_at_stop(step.from_stop_id)

        # Keep recommendations non-empty; if no directional match, use all available routes at origin.
        if not recommended_routes:
            recommended_routes = available_routes

        # If still empty, derive from graph edges (directional first, then any outgoing).
        if not recommended_routes:
            recommended_routes = _graph_routes(graph, step.from_stop_id, step.to_stop_id)
        if not recommended_routes:
            recommended_routes = _graph_routes(graph, step.from_stop_id)
        if not available_routes:
            available_routes = _graph_routes(graph, step.from_stop_id)

        if route_id is None and recommended_routes:
            route_id = recommended_routes[0]
        elif route_id is None and available_routes:
            route_id = available_routes[0]
    else:
        # Transfer edges: show all routes at the destination stop; also surface them as recommendations to board next.
        recommended_routes = engine.get_routes_at_stop(step.to_stop_id)
        available_routes = engine.get_routes_at_stop(step.to_stop_id)

        if not recommended_routes:
            recommended_routes = _graph_routes(graph, step.to_stop_id)
        if not available_routes:
            available_routes = recommended_routes

    return StepResponse(
        from_stop_id=step.from_stop_id,
        from_stop_name=from_attrs.get("stop_name"),
        to_stop_id=step.to_stop_id,
        to_stop_name=to_attrs.get("stop_name"),
        edge_type=step.edge_type,
        cost=step.cost,
        time_seconds=step.time_seconds,
        route_id=route_id,
        notes=step.attributes.get("notes"),
        to_coordinates=to_coordinates,
        available_routes=available_routes,
        recommended_routes=recommended_routes,
        is_transfer=is_transfer,
    )


def _coords_from_attrs(attrs: Dict[str, object]) -> Optional[Coordinate]:
    lat = to_optional_float(attrs.get("stop_lat"))
    lon = to_optional_float(attrs.get("stop_lon"))
    if lat is None or lon is None:
        return None
    return Coordinate(lat=lat, lon=lon)


def _graph_routes(
    graph: nx.MultiDiGraph, from_stop: str, to_stop: Optional[str] = None
) -> List[str]:
    """Derive routes from graph edges when CSV-based cache is empty.

    If to_stop is provided, only include edges that lead to that stop.
    Otherwise, include all outgoing edges from from_stop.
    """
    routes: set[str] = set()
    if from_stop not in graph:
        return []
    for _, neighbor, edge_data in graph.out_edges(from_stop, data=True):
        if to_stop and neighbor != to_stop:
            continue
        route_id = edge_data.get("route_id") or edge_data.get("name")
        if route_id is None:
            continue
        route_id_str = str(route_id).strip()
        if not route_id_str or route_id_str.lower() in {"nan", "none"}:
            continue
        routes.add(route_id_str)
    return sorted(routes)