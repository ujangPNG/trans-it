'use client';

import { useEffect, useRef } from 'react';
import { MapContainer, TileLayer, Marker, Popup, Polyline, useMap } from 'react-leaflet';
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
  notes?: string;
  toCoordinates?: Coordinate;
}

interface RouteResponse {
  totalFare: number;
  totalTimeSeconds: number;
  steps: StepResponse[];
  summary: string;
}

interface RouteMapProps {
  route: RouteResponse | null;
  origin?: { lat: number; lon: number };
  destination?: { lat: number; lon: number };
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

export default function RouteMap({ route, origin }: RouteMapProps) {
  const mapRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (mapRef.current && route) {
      gsap.fromTo(
        mapRef.current,
        { opacity: 0, scale: 0.95 },
        { opacity: 1, scale: 1, duration: 0.6, ease: 'power3.out' }
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

  // Add origin and destination if provided
  const allCoordinates = [...routeCoordinates];
  if (origin) {
    allCoordinates.unshift([origin.lat, origin.lon]);
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
    <div ref={mapRef} className="bg-white rounded-lg shadow-lg overflow-hidden">
      <div className="h-[500px] relative">
        <MapContainer
          center={defaultCenter}
          zoom={13}
          scrollWheelZoom={true}
          className="h-full w-full"
        >
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />

          {/* Origin marker */}
          {origin && (
            <Marker position={[origin.lat, origin.lon]} icon={startIcon}>
              <Popup>
                <strong>Titik Awal</strong>
                <br />
                {origin.lat.toFixed(6)}, {origin.lon.toFixed(6)}
              </Popup>
            </Marker>
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
                  icon={isLast ? endIcon : undefined}
                >
                  <Popup>
                    <strong>{step.toStopName || step.toStopId}</strong>
                    <br />
                    {step.toCoordinates.latitude.toFixed(6)},{' '}
                    {step.toCoordinates.longitude.toFixed(6)}
                    {step.routeId && (
                      <>
                        <br />
                        <span className="text-xs">Rute: {step.routeId}</span>
                      </>
                    )}
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
    </div>
  );
}
