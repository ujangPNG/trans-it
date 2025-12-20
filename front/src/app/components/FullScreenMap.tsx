'use client';

import { useEffect, useRef, useState } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Polyline, useMap, useMapEvents } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { gsap } from 'gsap';

// Fix Leaflet default icon issue with Next.js
if ('_getIconUrl' in L.Icon.Default.prototype) {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  delete (L.Icon.Default.prototype as any)._getIconUrl;
}
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
});

// Constants
const EDGE_TYPE_TRAVEL = 'travel';

interface Coordinate {
  latitude: number;
  longitude: number;
}

interface StepResponse {
  fromStopId: string;
  fromStopName?: string;
  toStopId: string;
  toStopName?: string;
  edgeType: string;
  cost: number;
  timeSeconds: number;
  routeId?: string;
  availableRoutes?: string[]; // NEW: All buses available at this stop
  notes?: string;
  toCoordinates?: Coordinate;
}

interface RouteResponse {
  totalFare: number;
  totalTimeSeconds: number;
  steps: StepResponse[];
  summary: string;
}

interface FullScreenMapProps {
  route: RouteResponse | null;
  originMarker: [number, number] | null;
  destinationMarker: [number, number] | null;
  onOriginChange: (lat: number, lon: number) => void;
  onDestinationChange: (lat: number, lon: number) => void;
  markerMode: 'origin' | 'destination' | null;
}

// Component to handle map clicks for placing markers
function MapClickHandler({ 
  onOriginChange, 
  onDestinationChange, 
  markerMode 
}: { 
  onOriginChange: (lat: number, lon: number) => void;
  onDestinationChange: (lat: number, lon: number) => void;
  markerMode: 'origin' | 'destination' | null;
}) {
  useMapEvents({
    click(e) {
      if (markerMode === 'origin') {
        onOriginChange(e.latlng.lat, e.latlng.lng);
      } else if (markerMode === 'destination') {
        onDestinationChange(e.latlng.lat, e.latlng.lng);
      }
    },
  });
  return null;
}

// Component to fit bounds when route changes
function FitBounds({ coordinates }: { coordinates: [number, number][] }) {
  const map = useMap();
  
  useEffect(() => {
    if (coordinates.length > 0) {
      const bounds = L.latLngBounds(coordinates);
      map.fitBounds(bounds, { padding: [50, 50] });
    }
  }, [coordinates, map]);

  return null;
}

// Custom draggable marker component
function DraggableMarker({
  position,
  onChange,
  icon,
  label,
}: {
  position: [number, number];
  onChange: (lat: number, lon: number) => void;
  icon: L.Icon;
  label: string;
}) {
  const markerRef = useRef<L.Marker>(null);

  const eventHandlers = {
    dragend() {
      const marker = markerRef.current;
      if (marker != null) {
        const pos = marker.getLatLng();
        onChange(pos.lat, pos.lng);
      }
    },
  };

  return (
    <Marker
      draggable={true}
      eventHandlers={eventHandlers}
      position={position}
      ref={markerRef}
      icon={icon}
    >
      <Popup>
        <strong>{label}</strong>
        <br />
        {position[0].toFixed(6)}, {position[1].toFixed(6)}
        <br />
        <small>Drag to move or use arrow keys</small>
      </Popup>
    </Marker>
  );
}

export default function FullScreenMap({
  route,
  originMarker,
  destinationMarker,
  onOriginChange,
  onDestinationChange,
  markerMode,
}: FullScreenMapProps) {
  const mapRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (mapRef.current && route) {
      gsap.fromTo(
        mapRef.current,
        { opacity: 0.8 },
        { opacity: 1, duration: 0.3 }
      );
    }
  }, [route]);

  // Default center (Jakarta)
  const defaultCenter: [number, number] = [-6.2088, 106.8456];

  // Extract coordinates from route steps
  const routeCoordinates: [number, number][] = [];
  if (route && route.steps.length > 0) {
    route.steps.forEach((step) => {
      if (step.toCoordinates) {
        routeCoordinates.push([
          step.toCoordinates.latitude,
          step.toCoordinates.longitude,
        ]);
      }
    });
  }

  // Collect all coordinates for bounds
  const allCoordinates = [...routeCoordinates];
  if (originMarker) {
    allCoordinates.unshift(originMarker);
  }
  if (destinationMarker && !routeCoordinates.length) {
    allCoordinates.push(destinationMarker);
  }

  // Custom icons
  const startIcon = new L.Icon({
    iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-green.png',
    shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
    iconSize: [25, 41],
    iconAnchor: [12, 41],
    popupAnchor: [1, -34],
    shadowSize: [41, 41],
  });

  const endIcon = new L.Icon({
    iconUrl: 'https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-red.png',
    shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
    iconSize: [25, 41],
    iconAnchor: [12, 41],
    popupAnchor: [1, -34],
    shadowSize: [41, 41],
  });

  return (
    <div ref={mapRef} className="h-full w-full">
      <MapContainer
        center={defaultCenter}
        zoom={13}
        scrollWheelZoom={true}
        className="h-full w-full"
        zoomControl={true}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {/* Map click handler for placing markers */}
        <MapClickHandler
          onOriginChange={onOriginChange}
          onDestinationChange={onDestinationChange}
          markerMode={markerMode}
        />

        {/* Origin marker (draggable) */}
        {originMarker && (
          <DraggableMarker
            position={originMarker}
            onChange={onOriginChange}
            icon={startIcon}
            label="Titik Awal"
          />
        )}

        {/* Destination marker (draggable) */}
        {destinationMarker && (
          <DraggableMarker
            position={destinationMarker}
            onChange={onDestinationChange}
            icon={endIcon}
            label="Tujuan"
          />
        )}

        {/* Route stops */}
        {route &&
          route.steps.map((step, index) => {
            if (!step.toCoordinates) return null;
            
            const isLast = index === route.steps.length - 1;
            return (
              <Marker
                key={index}
                position={[
                  step.toCoordinates.latitude,
                  step.toCoordinates.longitude,
                ]}
              >
                <Popup>
                  <div className="text-sm">
                    <strong className="text-base">{step.toStopName || step.toStopId}</strong>
                    <br />
                    <span className="text-xs text-gray-600">
                      {step.toCoordinates.latitude.toFixed(6)},{' '}
                      {step.toCoordinates.longitude.toFixed(6)}
                    </span>
                    {step.edgeType === EDGE_TYPE_TRAVEL ? (
                      step.routeId && (
                        <>
                          <br />
                          <span className="text-sm font-semibold text-blue-600">
                            🚌 Naik bus: {step.routeId}
                          </span>
                        </>
                      )
                    ) : (
                      <>
                        <br />
                        <span className="text-sm font-semibold text-orange-600">
                          🚶 Transfer
                        </span>
                      </>
                    )}
                    {step.edgeType === EDGE_TYPE_TRAVEL && step.availableRoutes && step.availableRoutes.length > 0 && (
                      <>
                        <br />
                        <span className="text-xs text-gray-500">
                          Bus tersedia: {step.availableRoutes.join(', ')}
                        </span>
                      </>
                    )}
                  </div>
                </Popup>
              </Marker>
            );
          })}

        {/* Route line */}
        {routeCoordinates.length > 0 && (
          <Polyline
            positions={routeCoordinates}
            color="#3b82f6"
            weight={4}
            opacity={0.7}
          />
        )}

        {/* Fit bounds to show all markers */}
        {allCoordinates.length > 0 && (
          <FitBounds coordinates={allCoordinates} />
        )}
      </MapContainer>
    </div>
  );
}
