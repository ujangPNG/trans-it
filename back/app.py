# FastAPI backend, port 25200.
# Basenya pake djikstra dari networkx, dengan modifikasi untuk fare-aware routing.
# source rute dari GTFS Transjakarta
# https://gtfs.transjakarta.co.id/files/file_gtfs.zip


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
    """Configuration settings for the transit routing engine.
    
    Contains paths to GTFS data files and parameters for fare calculation
    and transfer penalties.
    """
    # Path ke file stops.csv berisi daftar semua halte
    stops_path: Path = DEFAULT_DATA_DIR / "stops.csv"
    # Path ke file trips.csv berisi daftar trip dan route_id
    trips_path: Path = DEFAULT_DATA_DIR / "trips.csv"
    # Path ke file stop_times.csv berisi urutan halte dalam setiap trip
    stop_times_path: Path = DEFAULT_DATA_DIR / "stop_times.csv"
    # Path ke file data lokasi halte dengan koordinat
    halte_locations_path: Path = DEFAULT_HALTE_LOCATIONS_PATH
    # Penalty biaya untuk setiap perpindahan bus (dalam rupiah)
    fare_penalty: int = DEFAULT_FARE
    # Waktu transfer antar bus yang sama-sama TransJakarta (detik)
    internal_transfer_seconds: int = DEFAULT_INTERNAL_TRANSFER_SECONDS
    external_transfer_seconds: int = DEFAULT_EXTERNAL_TRANSFER_SECONDS
    default_travel_seconds: int = DEFAULT_TRAVEL_SECONDS
    manual_internal_groups_path: Optional[Path] = None
    external_transfer_config_path: Optional[Path] = None


def _path_from_env(var_name: str, default: Path) -> Path:
    """Ambil path dari environment variable atau gunakan default.
    
    Args:
        var_name: Nama environment variable
        default: Path default jika env var tidak ada
    
    Returns:
        Path dari env var atau default
    """
    value = os.getenv(var_name)
    if value:
        return Path(value)
    return default


def _optional_path_from_env(var_name: str) -> Optional[Path]:
    """Ambil path opsional dari environment variable.
    
    Args:
        var_name: Nama environment variable
    
    Returns:
        Path jika env var ada, None jika tidak
    """
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
    """Model untuk koordinat geografis (latitude, longitude)."""
    model_config = ConfigDict(populate_by_name=True)

    # Latitude (garis lintang)
    lat: float = Field(..., alias="latitude")
    # Longitude (garis bujur)
    lon: float = Field(..., alias="longitude")


class StepResponse(BaseModel):
    """Model response untuk satu langkah perjalanan dalam rute.
    
    Setiap step merepresentasikan perpindahan dari satu halte ke halte lain,
    baik dengan naik bus (travel) atau jalan kaki/transfer (transfer).
    """
    model_config = ConfigDict(populate_by_name=True)

    # ID halte asal
    from_stop_id: str = Field(..., alias="fromStopId")
    # Nama halte asal
    from_stop_name: Optional[str] = Field(None, alias="fromStopName")
    # ID halte tujuan
    to_stop_id: str = Field(..., alias="toStopId")
    # Nama halte tujuan
    to_stop_name: Optional[str] = Field(None, alias="toStopName")
    # Tipe edge: "travel" = naik bus, "internal-transfer" = pindah bus dalam sistem
    edge_type: str = Field(..., alias="edgeType")
    # Biaya langkah ini (0 untuk travel dalam satu bus, DEFAULT_FARE untuk transfer)
    cost: int
    # Estimasi waktu tempuh langkah ini (detik)
    time_seconds: float = Field(..., alias="timeSeconds")
    # ID rute bus yang direkomendasikan untuk langkah ini
    route_id: Optional[str] = Field(None, alias="routeId")
    # Catatan tambahan (misalnya instruksi khusus)
    notes: Optional[str] = None
    # Koordinat halte tujuan
    to_coordinates: Optional[Coordinate] = Field(None, alias="toCoordinates")
    # Semua bus yang tersedia di halte (untuk langkah travel: di halte asal, untuk transfer: di halte tujuan)
    available_routes: List[str] = Field(
        default_factory=list,
        alias="availableRoutes",
        description="All bus routes available at the relevant stop",
    )
    # Bus yang direkomendasikan: yang bisa langsung ke halte berikutnya (directional)
    recommended_routes: List[str] = Field(
        default_factory=list,
        alias="recommendedRoutes",
        description="Routes that directly serve the next stop in this step",
    )
    # True jika langkah ini adalah transfer (pindah bus), False jika naik bus
    is_transfer: bool = Field(False, alias="isTransfer", description="True when this step is a transfer between routes")


class RouteResponse(BaseModel):
    """Model response untuk hasil pencarian rute lengkap."""
    model_config = ConfigDict(populate_by_name=True)

    # Total biaya perjalanan (termasuk biaya transfer)
    total_fare: int = Field(..., alias="totalFare")
    # Total estimasi waktu tempuh (detik)
    total_time_seconds: float = Field(..., alias="totalTimeSeconds")
    # Daftar langkah-langkah perjalanan dari asal ke tujuan
    steps: List[StepResponse]
    # Ringkasan rute dalam format teks
    summary: str


class RouteRequest(BaseModel):
    """Model request untuk pencarian rute.
    
    Bisa menggunakan stop ID/name langsung, atau koordinat GPS untuk autoselection.
    """
    model_config = ConfigDict(populate_by_name=True)

    # ID atau nama halte asal (optional jika pakai lat/lon)
    origin: Optional[str] = Field(
        None, description="Stop ID or name for the origin halte"
    )
    # ID atau nama halte tujuan (optional jika pakai lat/lon)
    destination: Optional[str] = Field(
        None, description="Stop ID or name for the destination halte"
    )
    # Latitude halte asal untuk auto-select halte terdekat
    origin_lat: Optional[float] = Field(
        None, alias="originLat", description="Latitude for origin autoselection"
    )
    # Longitude halte asal untuk auto-select halte terdekat
    origin_lon: Optional[float] = Field(
        None, alias="originLon", description="Longitude for origin autoselection"
    )
    # Latitude halte tujuan untuk auto-select halte terdekat
    destination_lat: Optional[float] = Field(
        None, alias="destinationLat", description="Latitude for destination autoselection"
    )
    # Longitude halte tujuan untuk auto-select halte terdekat
    destination_lon: Optional[float] = Field(
        None, alias="destinationLon", description="Longitude for destination autoselection"
    )


class RoutingEngine:
    """Engine untuk mencari rute terbaik antar halte dengan graph-based routing.
    
    Menggunakan NetworkX MultiDiGraph untuk merepresentasikan jaringan transportasi.
    Mendukung transfer antar bus dengan fare penalty.
    """
    
    def __init__(self, settings: RouterSettings) -> None:
        """Inisialisasi routing engine dengan konfigurasi dari settings.
        
        Args:
            settings: Konfigurasi router (path file, penalty, dll)
        """
        self.settings = settings
        # Build graph dari GTFS data
        self.graph = self._build_graph()
        # Cache untuk mapping stop_id → routes
        self._routes_by_stop: Dict[str, List[str]] = {}
        # Build cache dari CSV
        self._build_routes_cache()

    def _build_graph(self) -> nx.MultiDiGraph:
        """Build graph transportasi dari data GTFS.
        
        Returns:
            NetworkX MultiDiGraph dengan nodes = halte, edges = koneksi bus/transfer
        """
        # Load manual groups untuk internal transfer jika ada
        manual_groups: Dict[str, List[str]] = {}
        if self.settings.manual_internal_groups_path and self.settings.manual_internal_groups_path.exists():
            manual_groups = load_manual_groups(str(self.settings.manual_internal_groups_path))

        # Load external transfers (antar sistem transportasi) jika ada
        external_transfers: List[Dict[str, object]] = []
        if self.settings.external_transfer_config_path and self.settings.external_transfer_config_path.exists():
            external_transfers = load_external_transfers(
                str(self.settings.external_transfer_config_path)
            )

        # Build graph menggunakan TransitGraphBuilder
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
        """Build cache mapping halte ke routes yang lewat sana dan koneksi antar halte.
        
        Cache yang dibangun:
        - _routes_by_stop: Dict[stop_id, List[route_id]] → semua bus di halte
        - _routes_between_stops: Dict[(from_stop, to_stop), set[route_id]] → bus yang langsung menghubungkan 2 halte
        """
        import csv
        from collections import defaultdict
        
        # Temporary sets untuk menghindari duplikasi
        routes_by_stop: Dict[str, set[str]] = defaultdict(set)
        # Mapping koneksi directional: (halte_asal, halte_tujuan) → set route_id
        self._routes_between_stops: Dict[Tuple[str, str], set[str]] = defaultdict(set)
        
        # Step 1: Baca trips.csv untuk mapping trip_id → route_id
        trip_to_route: Dict[str, str] = {}
        if self.settings.trips_path.exists():
            with self.settings.trips_path.open('r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    trip_id = row.get('trip_id', '').strip()
                    route_id = row.get('route_id', '').strip()
                    if trip_id and route_id:
                        trip_to_route[trip_id] = route_id
        
        # Step 2: Baca stop_times.csv untuk mapping stops ke routes dan urutan halte
        if self.settings.stop_times_path.exists():
            # Kelompokkan per trip_id untuk track urutan halte
            trip_stops: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
            
            with self.settings.stop_times_path.open('r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    trip_id = row.get('trip_id', '').strip()
                    stop_id = row.get('stop_id', '').strip()
                    stop_sequence = row.get('stop_sequence', '').strip()
                    
                    # Hanya proses jika trip_id valid dan ada di mapping
                    if trip_id in trip_to_route and stop_id:
                        route_id = trip_to_route[trip_id]
                        # Tambahkan route_id ke halte ini
                        routes_by_stop[stop_id].add(route_id)
                        
                        # Track urutan halte untuk trip ini
                        try:
                            seq = int(stop_sequence) if stop_sequence else 0
                            trip_stops[trip_id].append((seq, stop_id))
                        except ValueError:
                            pass
            
            # Step 3: Build koneksi stop-to-stop dari urutan halte
            for trip_id, stops_list in trip_stops.items():
                if trip_id not in trip_to_route:
                    continue
                    
                route_id = trip_to_route[trip_id]
                # Sort berdasarkan stop_sequence
                stops_list.sort(key=lambda x: x[0])
                
                # Buat edge untuk setiap pasangan halte berurutan
                for i in range(len(stops_list) - 1):
                    from_stop = stops_list[i][1]
                    to_stop = stops_list[i + 1][1]
                    # Simpan bahwa route_id ini menghubungkan from_stop → to_stop
                    self._routes_between_stops[(from_stop, to_stop)].add(route_id)
        
        # Konversi set ke sorted list untuk output yang konsisten
        self._routes_by_stop = {
            stop_id: sorted(routes) 
            for stop_id, routes in routes_by_stop.items()
        }

    def get_routes_at_stop(self, stop_id: str, next_stop_id: Optional[str] = None) -> List[str]:
        """Ambil daftar bus yang tersedia di halte.
        
        Args:
            stop_id: ID halte yang dicari
            next_stop_id: (Optional) ID halte tujuan berikutnya.
                         Jika diisi, hanya return bus yang langsung ke next_stop.
        
        Returns:
            List route_id (sorted) yang tersedia
        """
        if next_stop_id:
            # Mode directional: hanya bus yang langsung menghubungkan stop_id → next_stop_id
            routes = self._routes_between_stops.get((stop_id, next_stop_id), set())
            return sorted(routes)
        else:
            # Mode all: semua bus yang lewat halte ini
            return self._routes_by_stop.get(stop_id, [])

    def rebuild(self) -> None:
        """Rebuild graph dan cache dari awal (untuk hot reload)."""
        self.graph = self._build_graph()
        self._build_routes_cache()


@lru_cache(maxsize=1)
def get_engine() -> RoutingEngine:
    """Singleton factory untuk RoutingEngine (di-cache untuk reuse)."""
    return RoutingEngine(SETTINGS)


app = FastAPI(
    title="TransJakarta Fare-Aware Routing API",
    version="0.2.0",
    description="Backend facade over the fare-aware routing engine (port 25200).",
)

# API key untuk autentikasi request dari frontend
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY")


def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> str:
    """Verifikasi API key dari header request.

    Fungisinya untuk keamanan, biar ha ada yang asal request dari luar.
    
    Args:
        x_api_key: API key dari HTTP header X-API-Key
    
    Returns:
        API key jika valid
    
    Raises:
        HTTPException: 500 jika API key belum dikonfigurasi, 403 jika tidak match
    """
    if not INTERNAL_API_KEY:
        raise HTTPException(status_code=500, detail="API key not configured")
    if x_api_key != INTERNAL_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return x_api_key


@app.get("/health")
def healthcheck() -> Dict[str, object]:
    """Health check endpoint untuk monitoring status backend.
    
    Returns:
        Dictionary dengan status, jumlah halte (nodes), dan koneksi (edges)
    """
    engine = get_engine()
    return {
        "status": "ok",
        "stops": engine.graph.number_of_nodes(),
        "edges": engine.graph.number_of_edges(),
    }

@app.get("/")
def tes():
    """Endpoint test sederhana untuk cek API hidup."""
    return {
        "tes":"ya"
    }

@app.post("/route", response_model=RouteResponse)
def compute_route(
    payload: RouteRequest, api_key: str = Depends(verify_api_key)
) -> RouteResponse:
    """Endpoint utama untuk mencari rute terbaik antar halte.
    
    Args:
        payload: Request dengan origin/destination (ID atau lat/lon)
        api_key: API key tervalidasi (auto-injected via Depends)
    
    Returns:
        RouteResponse dengan steps, total cost, dan total time
    
    Raises:
        HTTPException: 400 jika origin/destination tidak lengkap, 404 jika tidak ditemukan rute
    """
    engine = get_engine()
    
    # Validasi: origin harus ada (baik stop ID atau koordinat)
    if payload.origin is None and payload.origin_lat is None and payload.origin_lon is None:
        raise HTTPException(status_code=400, detail="origin or originLat/originLon is required")
    
    # Validasi: destination harus ada (baik stop ID atau koordinat)
    if (
        payload.destination is None
        and payload.destination_lat is None
        and payload.destination_lon is None
    ):
        raise HTTPException(
            status_code=400, detail="destination or destinationLat/destinationLon is required"
        )

    # Build coordinate tuple jika ada lat/lon untuk origin
    source_coord = _build_optional_coordinate(payload.origin_lat, payload.origin_lon)
    # Build coordinate tuple jika ada lat/lon untuk destination
    dest_coord = _build_optional_coordinate(payload.destination_lat, payload.destination_lon)

    # Resolve stop ID (bisa dari nama/ID langsung atau auto-select dari koordinat)
    try:
        source_id = resolve_stop_id(engine.graph, payload.origin or "", coordinate=source_coord)
        target_id = resolve_stop_id(
            engine.graph, payload.destination or "", coordinate=dest_coord
        )
    except (KeyError, ValueError) as err:
        # Jika halte tidak ditemukan
        raise HTTPException(status_code=404, detail=str(err)) from err

    # Cari rute terbaik dengan Dijkstra yang fare-aware
    try:
        result = fare_priority_dijkstra(engine.graph, source_id, target_id)
    except ValueError as err:
        # Jika tidak ada path yang valid
        raise HTTPException(status_code=404, detail=str(err)) from err

    # Format hasil untuk response
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
    """Buat tuple koordinat dari lat/lon jika keduanya ada.
    
    Args:
        lat: Latitude (optional)
        lon: Longitude (optional)
    
    Returns:
        Tuple (lat, lon) atau None jika salah satu kosong
    
    Raises:
        HTTPException: 400 jika koordinat invalid
    """
    if lat is None and lon is None:
        return None
    try:
        return build_coordinate_tuple(lat, lon)
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err


def _step_to_response(step: PathStep, graph: nx.MultiDiGraph, engine: RoutingEngine) -> StepResponse:
    """Konversi PathStep internal ke StepResponse yang API-friendly.
    
    Fungsi ini melakukan banyak fallback untuk memastikan:
    - route_id selalu terisi (tidak null)
    - recommended_routes selalu ada (untuk panduan user naik bus apa)
    - available_routes menampilkan semua opsi di halte
    
    Args:
        step: PathStep dari hasil routing
        graph: NetworkX MultiDiGraph
        engine: RoutingEngine untuk akses cache routes
    
    Returns:
        StepResponse dengan semua field terisi
    """
    # Ambil atribut halte dari graph nodes
    from_attrs = graph.nodes.get(step.from_stop_id, {})
    to_attrs = graph.nodes.get(step.to_stop_id, {})
    to_coordinates = _coords_from_attrs(to_attrs)

    # Ekstrak route_id dari attributes edge, handle nilai NaN/None
    route_id = step.attributes.get("route_id") or step.attributes.get("name")
    if route_id is not None:
        try:
            # Konversi ke string dan validasi
            route_id_str = str(route_id)
            if route_id_str.lower() in ('nan', 'none', ''):
                route_id = None
            else:
                route_id = route_id_str
        except Exception:
            route_id = None

    # Tentukan apakah ini transfer atau travel
    is_transfer = step.edge_type != "travel"

    # LOGIC PENTING: Populate recommended_routes dan available_routes
    # Case 1: Untuk TRAVEL edges (naik bus dari halte A ke B)
    if not is_transfer:
        # recommended_routes: bus yang LANGSUNG dari from_stop ke to_stop (directional)
        recommended_routes = engine.get_routes_at_stop(step.from_stop_id, step.to_stop_id)
        # available_routes: SEMUA bus yang lewat from_stop
        available_routes = engine.get_routes_at_stop(step.from_stop_id)

        # Fallback 1: Jika directional cache kosong, gunakan semua bus di from_stop
        if not recommended_routes:
            recommended_routes = available_routes

        # Fallback 2: Jika masih kosong, cari dari graph edges (directional dulu)
        if not recommended_routes:
            recommended_routes = _graph_routes(graph, step.from_stop_id, step.to_stop_id)
        # Fallback 3: Jika masih kosong, ambil semua outgoing routes dari graph
        if not recommended_routes:
            recommended_routes = _graph_routes(graph, step.from_stop_id)
        # Fallback 4: Pastikan available_routes juga tidak kosong
        if not available_routes:
            available_routes = _graph_routes(graph, step.from_stop_id)

        # Fallback route_id: gunakan recommended pertama, atau available pertama
        if route_id is None and recommended_routes:
            route_id = recommended_routes[0]
        elif route_id is None and available_routes:
            route_id = available_routes[0]
    else:
        # Case 2: Untuk TRANSFER edges (pindah bus, jalan kaki, dll)
        # Tampilkan bus yang tersedia di halte TUJUAN (to_stop)
        recommended_routes = engine.get_routes_at_stop(step.to_stop_id)
        available_routes = engine.get_routes_at_stop(step.to_stop_id)

        # Fallback untuk transfer: cari dari graph jika cache kosong
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
    """Ekstrak koordinat dari atribut node graph.
    
    Args:
        attrs: Dictionary atribut node (dari graph.nodes[stop_id])
    
    Returns:
        Coordinate object atau None jika lat/lon tidak valid
    """
    lat = to_optional_float(attrs.get("stop_lat"))
    lon = to_optional_float(attrs.get("stop_lon"))
    if lat is None or lon is None:
        return None
    return Coordinate(lat=lat, lon=lon)


def _graph_routes(
    graph: nx.MultiDiGraph, from_stop: str, to_stop: Optional[str] = None
) -> List[str]:
    """Ekstrak route_id dari edges graph sebagai fallback.
    
    Fungsi ini digunakan ketika CSV cache tidak punya data.
    Memastikan user tetap dapat rekomendasi bus walaupun cache kosong.
    
    Args:
        graph: NetworkX MultiDiGraph
        from_stop: ID halte asal
        to_stop: (Optional) ID halte tujuan. Jika None, ambil semua outgoing routes.
    
    Returns:
        List route_id yang tersedia (sorted)
    """
    routes = set()
    
    if to_stop:
        # Mode directional: cari routes untuk edge spesifik from_stop → to_stop
        if graph.has_edge(from_stop, to_stop):
            # MultiDiGraph bisa punya multiple edges antar 2 node
            for edge_data in graph[from_stop][to_stop].values():
                rid = edge_data.get("route_id")
                if rid:
                    routes.add(rid)
    else:
        # Mode all outgoing: ambil semua routes dari edges keluar dari from_stop
        for neighbor in graph.successors(from_stop):
            for edge_data in graph[from_stop][neighbor].values():
                rid = edge_data.get("route_id")
                if rid:
                    routes.add(rid)
    
    return sorted(routes)