import csv
import os
from pathlib import Path

from dotenv import load_dotenv
from psycopg import connect
from psycopg.rows import dict_row

BASE_DIR = Path(__file__).resolve().parents[1] / "data" / "transjakarta"
CSV_FILES = {
    "routes": BASE_DIR / "routes.csv",
    "route_list": BASE_DIR / "route_list.csv",
    "trips": BASE_DIR / "trips.csv",
    "stops": BASE_DIR / "stops.csv",
    "stop_times": BASE_DIR / "stop_times.csv",
    "shapes": BASE_DIR / "shapes.csv",
    "frequencies": BASE_DIR / "frequencies.csv",
    "fare_rules": BASE_DIR / "fare_rules.csv",
    "halte_locations": BASE_DIR / "data-lokasi-halte.csv",
}
TABLE_DDLS = {
    "routes": """
        CREATE TABLE IF NOT EXISTS routes (
            route_id TEXT PRIMARY KEY,
            agency_id TEXT,
            route_short_name TEXT,
            route_long_name TEXT,
            route_desc TEXT,
            route_type INTEGER,
            route_url TEXT,
            route_color VARCHAR(6),
            route_text_color VARCHAR(6),
            route_sort_order INTEGER
        );
    """,
    "route_list": """
        CREATE TABLE IF NOT EXISTS route_list (
            id SERIAL PRIMARY KEY,
            route_id TEXT NOT NULL,
            trip_id TEXT NOT NULL,
            direction_id INTEGER,
            stop_headsign TEXT,
            start_time TEXT,
            end_time TEXT
        );
    """,
    "trips": """
        CREATE TABLE IF NOT EXISTS trips (
            trip_id TEXT PRIMARY KEY,
            route_id TEXT NOT NULL,
            service_id TEXT,
            trip_headsign TEXT,
            trip_short_name TEXT,
            direction_id INTEGER,
            block_id TEXT,
            shape_id TEXT,
            wheelchair_accessible INTEGER,
            bikes_allowed INTEGER
        );
    """,
    "stops": """
        CREATE TABLE IF NOT EXISTS stops (
            stop_id TEXT PRIMARY KEY,
            stop_code TEXT,
            stop_name TEXT,
            stop_desc TEXT,
            stop_lat DOUBLE PRECISION,
            stop_lon DOUBLE PRECISION,
            zone_id TEXT,
            stop_url TEXT,
            location_type INTEGER,
            wheelchair_boarding INTEGER,
            parent_station TEXT,
            platform_code TEXT
        );
    """,
    "stop_times": """
        CREATE TABLE IF NOT EXISTS stop_times (
            trip_id TEXT NOT NULL,
            stop_sequence INTEGER NOT NULL,
            stop_id TEXT,
            arrival_time TEXT,
            departure_time TEXT,
            stop_headsign TEXT,
            pickup_type INTEGER,
            drop_off_type INTEGER,
            continuous_pickup INTEGER,
            continuous_drop_off INTEGER,
            shape_dist_traveled DOUBLE PRECISION,
            timepoint INTEGER,
            PRIMARY KEY (trip_id, stop_sequence)
        );
    """,
    "shapes": """
        CREATE TABLE IF NOT EXISTS shapes (
            shape_id TEXT NOT NULL,
            shape_pt_sequence INTEGER NOT NULL,
            shape_pt_lat DOUBLE PRECISION,
            shape_pt_lon DOUBLE PRECISION,
            shape_dist_traveled DOUBLE PRECISION,
            PRIMARY KEY (shape_id, shape_pt_sequence)
        );
    """,
    "frequencies": """
        CREATE TABLE IF NOT EXISTS frequencies (
            trip_id TEXT NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT NOT NULL,
            headway_secs INTEGER,
            exact_times INTEGER,
            PRIMARY KEY (trip_id, start_time, end_time)
        );
    """,
    "fare_rules": """
        CREATE TABLE IF NOT EXISTS fare_rules (
            id SERIAL PRIMARY KEY,
            fare_id TEXT NOT NULL,
            route_id TEXT,
            origin_id TEXT,
            destination_id TEXT,
            contains_id TEXT
        );
    """,
    "halte_locations": """
        CREATE TABLE IF NOT EXISTS halte_locations (
            no_registrasi TEXT PRIMARY KEY,
            periode_data INTEGER,
            wilayah TEXT,
            jenis_halte TEXT,
            kode_barang TEXT,
            lokasi_alamat TEXT,
            lintang DOUBLE PRECISION,
            bujur DOUBLE PRECISION
        );
    """,
}
TABLE_COLUMNS = {
    "routes": [
        "route_id",
        "agency_id",
        "route_short_name",
        "route_long_name",
        "route_desc",
        "route_type",
        "route_url",
        "route_color",
        "route_text_color",
        "route_sort_order",
    ],
    "route_list": [
        "route_id",
        "trip_id",
        "direction_id",
        "stop_headsign",
        "start_time",
        "end_time",
    ],
    "trips": [
        "trip_id",
        "route_id",
        "service_id",
        "trip_headsign",
        "trip_short_name",
        "direction_id",
        "block_id",
        "shape_id",
        "wheelchair_accessible",
        "bikes_allowed",
    ],
    "stops": [
        "stop_id",
        "stop_code",
        "stop_name",
        "stop_desc",
        "stop_lat",
        "stop_lon",
        "zone_id",
        "stop_url",
        "location_type",
        "wheelchair_boarding",
        "parent_station",
        "platform_code",
    ],
    "stop_times": [
        "trip_id",
        "stop_sequence",
        "stop_id",
        "arrival_time",
        "departure_time",
        "stop_headsign",
        "pickup_type",
        "drop_off_type",
        "continuous_pickup",
        "continuous_drop_off",
        "shape_dist_traveled",
        "timepoint",
    ],
    "shapes": [
        "shape_id",
        "shape_pt_sequence",
        "shape_pt_lat",
        "shape_pt_lon",
        "shape_dist_traveled",
    ],
    "frequencies": [
        "trip_id",
        "start_time",
        "end_time",
        "headway_secs",
        "exact_times",
    ],
    "fare_rules": [
        "fare_id",
        "route_id",
        "origin_id",
        "destination_id",
        "contains_id",
    ],
    "halte_locations": [
        "periode_data",
        "wilayah",
        "no_registrasi",
        "jenis_halte",
        "kode_barang",
        "lokasi_alamat",
        "lintang",
        "bujur",
    ],
}

INSERT_QUERY = {
    "routes": """
        INSERT INTO routes (
            route_id, agency_id, route_short_name, route_long_name, route_desc,
            route_type, route_url, route_color, route_text_color, route_sort_order
        ) VALUES ({placeholders})
        ON CONFLICT (route_id) DO UPDATE SET
            agency_id = EXCLUDED.agency_id,
            route_short_name = EXCLUDED.route_short_name,
            route_long_name = EXCLUDED.route_long_name,
            route_desc = EXCLUDED.route_desc,
            route_type = EXCLUDED.route_type,
            route_url = EXCLUDED.route_url,
            route_color = EXCLUDED.route_color,
            route_text_color = EXCLUDED.route_text_color,
            route_sort_order = EXCLUDED.route_sort_order;
    """,
    "route_list": """
        INSERT INTO route_list (
            route_id, trip_id, direction_id, stop_headsign, start_time, end_time
        ) VALUES ({placeholders});
    """,
    "trips": """
        INSERT INTO trips (
            trip_id, route_id, service_id, trip_headsign, trip_short_name,
            direction_id, block_id, shape_id, wheelchair_accessible, bikes_allowed
        ) VALUES ({placeholders})
        ON CONFLICT (trip_id) DO UPDATE SET
            route_id = EXCLUDED.route_id,
            service_id = EXCLUDED.service_id,
            trip_headsign = EXCLUDED.trip_headsign,
            trip_short_name = EXCLUDED.trip_short_name,
            direction_id = EXCLUDED.direction_id,
            block_id = EXCLUDED.block_id,
            shape_id = EXCLUDED.shape_id,
            wheelchair_accessible = EXCLUDED.wheelchair_accessible,
            bikes_allowed = EXCLUDED.bikes_allowed;
    """,
    "stops": """
        INSERT INTO stops (
            stop_id, stop_code, stop_name, stop_desc, stop_lat, stop_lon,
            zone_id, stop_url, location_type, wheelchair_boarding,
            parent_station, platform_code
        ) VALUES ({placeholders})
        ON CONFLICT (stop_id) DO UPDATE SET
            stop_code = EXCLUDED.stop_code,
            stop_name = EXCLUDED.stop_name,
            stop_desc = EXCLUDED.stop_desc,
            stop_lat = EXCLUDED.stop_lat,
            stop_lon = EXCLUDED.stop_lon,
            zone_id = EXCLUDED.zone_id,
            stop_url = EXCLUDED.stop_url,
            location_type = EXCLUDED.location_type,
            wheelchair_boarding = EXCLUDED.wheelchair_boarding,
            parent_station = EXCLUDED.parent_station,
            platform_code = EXCLUDED.platform_code;
    """,
    "stop_times": """
        INSERT INTO stop_times (
            trip_id, stop_sequence, stop_id, arrival_time, departure_time,
            stop_headsign, pickup_type, drop_off_type, continuous_pickup,
            continuous_drop_off, shape_dist_traveled, timepoint
        ) VALUES ({placeholders})
        ON CONFLICT (trip_id, stop_sequence) DO UPDATE SET
            stop_id = EXCLUDED.stop_id,
            arrival_time = EXCLUDED.arrival_time,
            departure_time = EXCLUDED.departure_time,
            stop_headsign = EXCLUDED.stop_headsign,
            pickup_type = EXCLUDED.pickup_type,
            drop_off_type = EXCLUDED.drop_off_type,
            continuous_pickup = EXCLUDED.continuous_pickup,
            continuous_drop_off = EXCLUDED.continuous_drop_off,
            shape_dist_traveled = EXCLUDED.shape_dist_traveled,
            timepoint = EXCLUDED.timepoint;
    """,
    "shapes": """
        INSERT INTO shapes (
            shape_id, shape_pt_sequence, shape_pt_lat, shape_pt_lon, shape_dist_traveled
        ) VALUES ({placeholders})
        ON CONFLICT (shape_id, shape_pt_sequence) DO UPDATE SET
            shape_pt_lat = EXCLUDED.shape_pt_lat,
            shape_pt_lon = EXCLUDED.shape_pt_lon,
            shape_dist_traveled = EXCLUDED.shape_dist_traveled;
    """,
    "frequencies": """
        INSERT INTO frequencies (
            trip_id, start_time, end_time, headway_secs, exact_times
        ) VALUES ({placeholders})
        ON CONFLICT (trip_id, start_time, end_time) DO UPDATE SET
            headway_secs = EXCLUDED.headway_secs,
            exact_times = EXCLUDED.exact_times;
    """,
    "fare_rules": """
        INSERT INTO fare_rules (
            fare_id, route_id, origin_id, destination_id, contains_id
        ) VALUES ({placeholders});
    """,
    "halte_locations": """
        INSERT INTO halte_locations (
            periode_data, wilayah, no_registrasi, jenis_halte, kode_barang,
            lokasi_alamat, lintang, bujur
        ) VALUES ({placeholders})
        ON CONFLICT (no_registrasi) DO UPDATE SET
            periode_data = EXCLUDED.periode_data,
            wilayah = EXCLUDED.wilayah,
            jenis_halte = EXCLUDED.jenis_halte,
            kode_barang = EXCLUDED.kode_barang,
            lokasi_alamat = EXCLUDED.lokasi_alamat,
            lintang = EXCLUDED.lintang,
            bujur = EXCLUDED.bujur;
    """,
}


def ensure_schema(conn):
    with conn.cursor() as cur:
        for ddl in TABLE_DDLS.values():
            cur.execute(ddl)
    conn.commit()


def load_env():
    if not os.environ.get("DATABASE_URL"):
        load_dotenv(Path(__file__).resolve().parents[1] / "front" / ".env")


def read_rows(path: Path, columns):
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        if reader.fieldnames is None:
            return
        reader.fieldnames = [h.strip() if h else "" for h in reader.fieldnames]
        for row in reader:
            normalized = {}
            for key, value in row.items():
                norm_key = key.strip() if isinstance(key, str) else key
                if isinstance(value, str):
                    value = value.strip()
                    if value == "":
                        value = None
                normalized[norm_key] = value
            if not any(normalized.get(column) is not None for column in columns):
                continue
            yield [normalized.get(column) for column in columns]


def truncate_tables(conn):
    with conn.cursor() as cur:
        cur.execute(
            "TRUNCATE TABLE route_list, stop_times, shapes, frequencies, fare_rules RESTART IDENTITY CASCADE;"
        )
        cur.execute(
            "TRUNCATE TABLE halte_locations, trips, stops, routes RESTART IDENTITY CASCADE;"
        )
        conn.commit()


def import_table(conn, table_name):
    path = CSV_FILES[table_name]
    if not path.exists():
        print(f"Skipping {table_name}: {path} missing")
        return 0

    columns = TABLE_COLUMNS[table_name]
    placeholders = ", ".join([f"%s" for _ in columns])
    query = INSERT_QUERY[table_name].format(placeholders=placeholders)
    rows = list(read_rows(path, columns))
    if not rows:
        print(f"No data for {table_name}")
        return 0

    with conn.cursor() as cur:
        cur.executemany(query, rows)
    conn.commit()
    print(f"Inserted {len(rows)} rows into {table_name}")
    return len(rows)


def main():
    load_env()
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL is not set")

    with connect(dsn, row_factory=dict_row) as conn:
        ensure_schema(conn)
        truncate_tables(conn)
        total = 0
        for table_name in CSV_FILES.keys():
            total += import_table(conn, table_name)
        print(f"Import completed. Total rows inserted: {total}")


if __name__ == "__main__":
    main()
