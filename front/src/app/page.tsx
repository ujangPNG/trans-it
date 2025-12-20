'use client';

import { useState, useEffect, useRef } from 'react';
import dynamic from 'next/dynamic';
import { gsap } from 'gsap';

// Dynamically import FullScreenMap to avoid SSR issues with Leaflet
const FullScreenMap = dynamic(() => import('./components/FullScreenMap'), {
  ssr: false,
  loading: () => (
    <div className="h-full w-full flex items-center justify-center bg-gray-200">
      <p className="text-gray-500">Loading map...</p>
    </div>
  ),
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

export default function Home() {
  const [route, setRoute] = useState<RouteResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [originMarker, setOriginMarker] = useState<[number, number] | null>(null);
  const [destinationMarker, setDestinationMarker] = useState<[number, number] | null>(null);
  const [markerMode, setMarkerMode] = useState<'origin' | 'destination' | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false); // Start closed
  
  const sidebarRef = useRef<HTMLDivElement>(null);
  const searchBarRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Animate search bar on mount
    if (searchBarRef.current) {
      gsap.fromTo(
        searchBarRef.current,
        { opacity: 0, y: -20 },
        { opacity: 1, y: 0, duration: 0.5, ease: 'power3.out' }
      );
    }
  }, []);

  const handleSearch = async () => {
    if (!originMarker || !destinationMarker) {
      setError('Please select both origin and destination on the map');
      return;
    }

    setIsLoading(true);
    setError(null);
    setRoute(null);

    try {
      const response = await fetch('/api/route', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          originLat: originMarker[0],
          originLon: originMarker[1],
          destinationLat: destinationMarker[0],
          destinationLon: destinationMarker[1],
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to fetch route');
      }

      const routeData: RouteResponse = await response.json();
      setRoute(routeData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  const handleOriginChange = (lat: number, lon: number) => {
    setOriginMarker([lat, lon]);
    setMarkerMode(null);
  };

  const handleDestinationChange = (lat: number, lon: number) => {
    setDestinationMarker([lat, lon]);
    setMarkerMode(null);
  };

  const getCurrentLocation = (isOrigin: boolean) => {
    if (!navigator.geolocation) {
      setError('Geolocation is not supported by your browser');
      return;
    }

    navigator.geolocation.getCurrentPosition(
      (position) => {
        const coords: [number, number] = [
          position.coords.latitude,
          position.coords.longitude,
        ];
        if (isOrigin) {
          setOriginMarker(coords);
        } else {
          setDestinationMarker(coords);
        }
      },
      (err) => {
        setError('Unable to retrieve your location: ' + err.message);
      }
    );
  };

  const formatTime = (seconds: number) => {
    const minutes = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${minutes} mnt ${secs > 0 ? `${secs} dtk` : ''}`;
  };

  const formatCurrency = (amount: number) => {
    return new Intl.NumberFormat('id-ID', {
      style: 'currency',
      currency: 'IDR',
      minimumFractionDigits: 0,
    }).format(amount);
  };

  const toggleSidebar = () => {
    if (sidebarRef.current) {
      if (sidebarOpen) {
        gsap.to(sidebarRef.current, {
          x: -360,
          duration: 0.3,
          ease: 'power2.inOut',
        });
      } else {
        gsap.to(sidebarRef.current, {
          x: 0,
          duration: 0.3,
          ease: 'power2.inOut',
        });
      }
    }
    setSidebarOpen(!sidebarOpen);
  };

  const handleSearchBarClick = () => {
    if (!sidebarOpen) {
      toggleSidebar();
    }
  };

  return (
    <div className="h-screen w-screen overflow-hidden flex relative">
      {/* Top Search Bar - "Mau ke mana?" */}
      <div
        ref={searchBarRef}
        className="absolute top-4 left-1/2 transform -translate-x-1/2 z-[1001] w-full max-w-md px-4"
      >
        <button
          onClick={handleSearchBarClick}
          className="w-full bg-white shadow-lg rounded-full px-6 py-4 flex items-center gap-3 hover:shadow-xl transition-shadow"
        >
          <svg
            className="w-6 h-6 text-gray-400"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
            />
          </svg>
          <span className="text-gray-600 font-medium flex-1 text-left">
            Mau ke mana?
          </span>
        </button>
      </div>

      {/* Sidebar */}
      <div
        ref={sidebarRef}
        className="absolute left-0 top-0 h-full w-[360px] bg-white shadow-2xl z-[1000] overflow-y-auto"
        style={{ transform: 'translateX(-360px)' }}
      >
        <div className="p-4 space-y-4">
          {/* Header */}
          <div className="flex items-center justify-between mb-4">
            <h1 className="text-xl font-bold text-gray-800">
              🚌 TransJakarta
            </h1>
            <button
              onClick={toggleSidebar}
              className="p-2 hover:bg-gray-100 rounded-lg"
              aria-label="Close sidebar"
            >
              ✕
            </button>
          </div>

          {/* Location Selection */}
          <div className="space-y-3">
            <div className="bg-gray-50 rounded-lg p-3">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                📍 Titik Awal
              </label>
              {originMarker ? (
                <div className="text-xs text-gray-600 mb-2">
                  {originMarker[0].toFixed(6)}, {originMarker[1].toFixed(6)}
                </div>
              ) : (
                <div className="text-xs text-gray-400 mb-2">
                  Belum dipilih
                </div>
              )}
              <div className="flex gap-2">
                <button
                  onClick={() => setMarkerMode('origin')}
                  className={`flex-1 px-3 py-2 text-xs rounded-md font-medium transition-colors ${
                    markerMode === 'origin'
                      ? 'bg-green-600 text-white'
                      : 'bg-green-100 text-green-700 hover:bg-green-200'
                  }`}
                >
                  {markerMode === 'origin' ? 'Klik peta...' : 'Pilih di peta'}
                </button>
                <button
                  onClick={() => getCurrentLocation(true)}
                  className="px-3 py-2 text-xs bg-blue-100 text-blue-700 hover:bg-blue-200 rounded-md font-medium"
                >
                  Lokasi saya
                </button>
              </div>
            </div>

            <div className="bg-gray-50 rounded-lg p-3">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                🎯 Tujuan
              </label>
              {destinationMarker ? (
                <div className="text-xs text-gray-600 mb-2">
                  {destinationMarker[0].toFixed(6)}, {destinationMarker[1].toFixed(6)}
                </div>
              ) : (
                <div className="text-xs text-gray-400 mb-2">
                  Belum dipilih
                </div>
              )}
              <div className="flex gap-2">
                <button
                  onClick={() => setMarkerMode('destination')}
                  className={`flex-1 px-3 py-2 text-xs rounded-md font-medium transition-colors ${
                    markerMode === 'destination'
                      ? 'bg-red-600 text-white'
                      : 'bg-red-100 text-red-700 hover:bg-red-200'
                  }`}
                >
                  {markerMode === 'destination' ? 'Klik peta...' : 'Pilih di peta'}
                </button>
                <button
                  onClick={() => getCurrentLocation(false)}
                  className="px-3 py-2 text-xs bg-blue-100 text-blue-700 hover:bg-blue-200 rounded-md font-medium"
                >
                  Lokasi saya
                </button>
              </div>
            </div>
          </div>

          {/* Search Button */}
          <button
            onClick={handleSearch}
            disabled={isLoading || !originMarker || !destinationMarker}
            className={`w-full py-3 px-4 rounded-lg font-semibold text-white transition-colors ${
              isLoading || !originMarker || !destinationMarker
                ? 'bg-gray-400 cursor-not-allowed'
                : 'bg-blue-600 hover:bg-blue-700'
            }`}
          >
            {isLoading ? 'Mencari rute...' : 'Cari Rute Termurah'}
          </button>

          {/* Error Display */}
          {error && (
            <div className="bg-red-50 border border-red-300 rounded-lg p-3">
              <p className="text-sm text-red-700">{error}</p>
            </div>
          )}

          {/* Route Results */}
          {route && (
            <div className="space-y-3">
              {/* Summary */}
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <p className="text-xs text-gray-600">Total Biaya</p>
                    <p className="text-lg font-bold text-blue-700">
                      {formatCurrency(route.totalFare)}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-gray-600">Waktu Tempuh</p>
                    <p className="text-lg font-bold text-blue-700">
                      {formatTime(route.totalTimeSeconds)}
                    </p>
                  </div>
                </div>
              </div>

              {/* Steps */}
              <div className="space-y-2">
                <h3 className="text-sm font-semibold text-gray-700">
                  Langkah Perjalanan ({route.steps.length})
                </h3>
                <div className="max-h-[calc(100vh-500px)] overflow-y-auto space-y-2">
                  {route.steps.map((step, index) => (
                    <div
                      key={index}
                      className="bg-gray-50 border border-gray-200 rounded-lg p-3 text-xs"
                    >
                      <div className="flex items-start gap-2">
                        <div className="flex-shrink-0">
                          <div className="w-6 h-6 bg-blue-600 text-white rounded-full flex items-center justify-center font-bold text-[10px]">
                            {index + 1}
                          </div>
                        </div>
                        <div className="flex-grow">
                          <p className="font-semibold text-gray-800">
                            {step.fromStopName || step.fromStopId}
                          </p>
                          <p className="text-gray-400 my-1">↓</p>
                          <p className="font-semibold text-gray-800">
                            {step.toStopName || step.toStopId}
                          </p>
                          <div className="mt-2 flex items-center gap-2 text-[10px] text-gray-500">
                            <span>⏱ {formatTime(step.timeSeconds)}</span>
                            {step.cost > 0 && (
                              <span className="text-green-600 font-semibold">
                                💰 {formatCurrency(step.cost)}
                              </span>
                            )}
                          </div>
                          {step.routeId && (
                            <div className="mt-2">
                              <div className="text-[10px] bg-blue-100 text-blue-800 px-2 py-1 rounded inline-block">
                                Naik: {step.routeId}
                              </div>
                            </div>
                          )}
                          {step.availableRoutes && step.availableRoutes.length > 0 && (
                            <div className="mt-2">
                              <p className="text-[9px] text-gray-500 mb-1">Bus tersedia di halte:</p>
                              <div className="flex flex-wrap gap-1">
                                {step.availableRoutes.map((route, idx) => (
                                  <span
                                    key={idx}
                                    className={`text-[9px] px-1.5 py-0.5 rounded ${
                                      route === step.routeId
                                        ? 'bg-blue-600 text-white font-semibold'
                                        : 'bg-gray-200 text-gray-700'
                                    }`}
                                  >
                                    {route}
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Info */}
          <div className="text-xs text-gray-500 pt-4 border-t border-gray-200">
            <p className="mb-2">💡 Cara menggunakan:</p>
            <ul className="list-disc list-inside space-y-1 text-[11px]">
              <li>Klik "Mau ke mana?" di atas untuk mulai</li>
              <li>Klik "Pilih di peta" lalu klik lokasi di peta</li>
              <li>Atau gunakan "Lokasi saya" untuk posisi saat ini</li>
              <li>Marker bisa di-drag untuk menyesuaikan posisi</li>
              <li>Klik "Cari Rute Termurah" untuk hasil</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Full Screen Map */}
      <div className="flex-1 h-full w-full">
        <FullScreenMap
          route={route}
          originMarker={originMarker}
          destinationMarker={destinationMarker}
          onOriginChange={handleOriginChange}
          onDestinationChange={handleDestinationChange}
          markerMode={markerMode}
        />
      </div>
    </div>
  );
}
