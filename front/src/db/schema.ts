import { primaryKey } from "drizzle-orm/pg-core";
import {
  doublePrecision,
  integer,
  pgTable,
  serial,
  text,
  varchar,
} from "drizzle-orm/pg-core";

export const routes = pgTable("routes", {
  routeId: text("route_id").primaryKey(),
  agencyId: text("agency_id"),
  routeShortName: text("route_short_name"),
  routeLongName: text("route_long_name"),
  routeDesc: text("route_desc"),
  routeType: integer("route_type"),
  routeUrl: text("route_url"),
  routeColor: varchar("route_color", { length: 6 }),
  routeTextColor: varchar("route_text_color", { length: 6 }),
  routeSortOrder: integer("route_sort_order"),
});

export const routesBusOnly = pgTable("routes_bus_only", {
  routeId: text("route_id").primaryKey(),
  agencyId: text("agency_id"),
  routeShortName: text("route_short_name"),
  routeLongName: text("route_long_name"),
  routeDesc: text("route_desc"),
  routeType: integer("route_type"),
  routeUrl: text("route_url"),
  routeColor: varchar("route_color", { length: 6 }),
  routeTextColor: varchar("route_text_color", { length: 6 }),
  routeSortOrder: integer("route_sort_order"),
});

export const routeList = pgTable("route_list", {
  id: serial("id").primaryKey(),
  routeId: text("route_id").notNull(),
  tripId: text("trip_id").notNull(),
  directionId: integer("direction_id"),
  stopHeadsign: text("stop_headsign"),
  startTime: text("start_time"),
  endTime: text("end_time"),
});

export const trips = pgTable("trips", {
  tripId: text("trip_id").primaryKey(),
  routeId: text("route_id").notNull(),
  serviceId: text("service_id"),
  tripHeadsign: text("trip_headsign"),
  tripShortName: text("trip_short_name"),
  directionId: integer("direction_id"),
  blockId: text("block_id"),
  shapeId: text("shape_id"),
  wheelchairAccessible: integer("wheelchair_accessible"),
  bikesAllowed: integer("bikes_allowed"),
});

export const stops = pgTable("stops", {
  stopId: text("stop_id").primaryKey(),
  stopCode: text("stop_code"),
  stopName: text("stop_name"),
  stopDesc: text("stop_desc"),
  stopLat: doublePrecision("stop_lat"),
  stopLon: doublePrecision("stop_lon"),
  zoneId: text("zone_id"),
  stopUrl: text("stop_url"),
  locationType: integer("location_type"),
  wheelchairBoarding: integer("wheelchair_boarding"),
  parentStation: text("parent_station"),
  platformCode: text("platform_code"),
});

export const stopTimes = pgTable(
  "stop_times",
  {
    tripId: text("trip_id").notNull(),
    stopSequence: integer("stop_sequence").notNull(),
    stopId: text("stop_id"),
    arrivalTime: text("arrival_time"),
    departureTime: text("departure_time"),
    stopHeadsign: text("stop_headsign"),
    pickupType: integer("pickup_type"),
    dropOffType: integer("drop_off_type"),
    continuousPickup: integer("continuous_pickup"),
    continuousDropOff: integer("continuous_drop_off"),
    shapeDistTraveled: doublePrecision("shape_dist_traveled"),
    timepoint: integer("timepoint"),
  },
  (table) => ({
    pk: primaryKey({
      name: "stop_times_pk",
      columns: [table.tripId, table.stopSequence],
    }),
  })
);

export const shapes = pgTable(
  "shapes",
  {
    shapeId: text("shape_id").notNull(),
    shapePtSequence: integer("shape_pt_sequence").notNull(),
    shapePtLat: doublePrecision("shape_pt_lat"),
    shapePtLon: doublePrecision("shape_pt_lon"),
    shapeDistTraveled: doublePrecision("shape_dist_traveled"),
  },
  (table) => ({
    pk: primaryKey({
      name: "shapes_pk",
      columns: [table.shapeId, table.shapePtSequence],
    }),
  })
);

export const frequencies = pgTable(
  "frequencies",
  {
    tripId: text("trip_id").notNull(),
    startTime: text("start_time").notNull(),
    endTime: text("end_time").notNull(),
    headwaySecs: integer("headway_secs"),
    exactTimes: integer("exact_times"),
  },
  (table) => ({
    pk: primaryKey({
      name: "frequencies_pk",
      columns: [table.tripId, table.startTime, table.endTime],
    }),
  })
);

export const fareRules = pgTable("fare_rules", {
  id: serial("id").primaryKey(),
  fareId: text("fare_id").notNull(),
  routeId: text("route_id"),
  originId: text("origin_id"),
  destinationId: text("destination_id"),
  containsId: text("contains_id"),
});

export const halteLocations = pgTable("halte_locations", {
  registrationNumber: text("no_registrasi").primaryKey(),
  periodeData: integer("periode_data"),
  region: text("wilayah"),
  halteType: text("jenis_halte"),
  assetCode: text("kode_barang"),
  address: text("lokasi_alamat"),
  latitude: doublePrecision("lintang"),
  longitude: doublePrecision("bujur"),
});

