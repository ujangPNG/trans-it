'use client';

import { useState, useEffect, useRef } from 'react';
import dynamic from 'next/dynamic';
import { gsap } from 'gsap';

// Dynamically import FullScreenMap to avoid SSR issues with Leaflet
const FullScreenMap = dynamic(() => import('./components/FullScreenMap'), {
  ssr: false,
  loading: () => (
    <div className="h-full w-full flex items-center justify-center bg-gray-100">
      <div className="flex flex-col items-center gap-2">
        <div className="w-12 h-12 border-4 border-[#FFC107] border-t-transparent rounded-full animate-spin"></div>
        <p className="text-gray-600">Memuat peta...</p>
      </div>
    </div>
  ),
});

// Constants
const EDGE_TYPE_TRAVEL = 'travel';

interface Coordinate {
  latitude: number;
  longitude: number;
}

interface RouteStep {
  fromStopId: string;
  fromStopName: string;
  toStopId: string;
  toStopName: string;
  edgeType: string;
  cost: number;
  timeSeconds: number;
  routeId?: string | null;
  notes?: string | null;
  toCoordinates: Coordinate;
  availableRoutes?: string[];
}

interface RouteResponse {
  totalFare: number;
  totalTimeSeconds: number;
  steps: RouteStep[];
  summary: string;
}

export default function Home() {
  const [origin, setOrigin] = useState<Coordinate | null>(null);
  const [destination, setDestination] = useState<Coordinate | null>(null);
  const [route, setRoute] = useState<RouteResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [pickingOrigin, setPickingOrigin] = useState(false);
  const [pickingDestination, setPickingDestination] = useState(false);

  const sidebarRef = useRef<HTMLDivElement>(null);
  const searchBarRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Initialize sidebar position (closed)
    if (sidebarRef.current) {
      gsap.set(sidebarRef.current, { x: -400 });
    }
  }, []);

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    if (mins > 0) {
      return `${mins} menit${secs > 0 ? ` ${secs} detik` : ''}`;
    }
    return `${secs} detik`;
  };

  const formatCurrency = (amount: number): string => {
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
          x: -400,
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

  const handleMapClick = (lat: number, lng: number) => {
    if (pickingOrigin) {
      setOrigin({ latitude: lat, longitude: lng });
      setPickingOrigin(false);
    } else if (pickingDestination) {
      setDestination({ latitude: lat, longitude: lng });
      setPickingDestination(false);
    }
  };

  const handleOriginDrag = (lat: number, lng: number) => {
    setOrigin({ latitude: lat, longitude: lng });
  };

  const handleDestinationDrag = (lat: number, lng: number) => {
    setDestination({ latitude: lat, longitude: lng });
  };

  const getCurrentLocation = (forOrigin: boolean) => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const coord: Coordinate = {
            latitude: position.coords.latitude,
            longitude: position.coords.longitude,
          };
          if (forOrigin) {
            setOrigin(coord);
          } else {
            setDestination(coord);
          }
        },
        (error) => {
          console.error('Error getting location:', error);
          setError('Tidak dapat mengakses lokasi. Pastikan izin lokasi diaktifkan.');
        }
      );
    } else {
      setError('Geolocation tidak didukung oleh browser Anda.');
    }
  };

  const searchRoute = async () => {
    if (!origin || !destination) {
      setError('Mohon pilih titik awal dan tujuan');
      return;
    }

    setLoading(true);
    setError(null);
    setRoute(null);

    try {
      const response = await fetch('/api/route', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          originLat: origin.latitude,
          originLon: origin.longitude,
          destinationLat: destination.latitude,
          destinationLon: destination.longitude,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Gagal mencari rute');
      }

      const data: RouteResponse = await response.json();
      setRoute(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Terjadi kesalahan');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="h-screen w-screen overflow-hidden flex relative bg-gray-100">
      {/* Top Search Bar - "Mau ke mana?" */}
      <div
        ref={searchBarRef}
        className="absolute top-4 left-1/2 transform -translate-x-1/2 z-1001 w-full max-w-md px-4"
      >
        <button
          onClick={handleSearchBarClick}
          className="w-full bg-white shadow-lg rounded-lg px-5 py-3.5 flex items-center gap-3 hover:shadow-xl transition-all border border-gray-200"
        >
          <svg
            className="w-5 h-5 text-gray-500"
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
          <span className="text-gray-700 font-normal flex-1 text-left text-sm">
            Mau ke mana?
          </span>
        </button>
      </div>

      {/* Sidebar */}
      <div
        ref={sidebarRef}
        className="absolute left-0 top-0 h-full w-[400px] bg-white shadow-xl z-1000 overflow-y-auto"
        style={{ transform: 'translateX(-400px)' }}
      >
        <div className="flex flex-col h-full">
          {/* Header */}
          <div className="bg-[#1a1a1a] text-white p-4">
            <div className="flex items-center justify-between mb-3">
              <h1 className="text-lg font-semibold flex items-center gap-2">
                <span>TransIt</span>
              </h1>
              <button
                onClick={toggleSidebar}
                className="p-2 hover:bg-[#2d2d2d] rounded-lg transition-colors"
                aria-label="Close sidebar"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <p className="text-xs text-gray-300">Cari rute termurah TransJakarta</p>
          </div>

          {/* Content */}
          <div className="flex-1 p-4 space-y-4">
            {/* Origin */}
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-sm font-medium text-gray-700">
                <div className="w-3 h-3 rounded-full bg-green-500"></div>
                <span>Titik Awal</span>
              </div>
              <div className="bg-gray-50 rounded-lg p-3 border border-gray-200">
                {origin ? (
                  <div className="text-xs text-gray-600">
                    <div className="font-mono">{origin.latitude.toFixed(6)}, {origin.longitude.toFixed(6)}</div>
                  </div>
                ) : (
                  <div className="text-xs text-gray-400">Belum dipilih</div>
                )}
                <div className="flex gap-2 mt-2">
                  <button
                    onClick={() => {
                      setPickingOrigin(!pickingOrigin);
                      setPickingDestination(false);
                    }}
                    className={`flex-1 px-3 py-2 text-xs rounded-md font-medium transition-colors ${
                      pickingOrigin
                        ? 'bg-green-500 text-white'
                        : 'bg-white border border-gray-300 text-gray-700 hover:bg-gray-50'
                    }`}
                  >
                    {pickingOrigin ? 'Klik peta...' : 'Pilih di peta'}
                  </button>
                  <button
                    onClick={() => getCurrentLocation(true)}
                    className="px-3 py-2 text-xs bg-white border border-gray-300 text-gray-700 rounded-md hover:bg-gray-50 transition-colors font-medium"
                  >
                    Lokasi saya
                  </button>
                </div>
              </div>
            </div>

            {/* Destination */}
            <div className="space-y-2">
              <div className="flex items-center gap-2 text-sm font-medium text-gray-700">
                <div className="w-3 h-3 rounded-full bg-red-500"></div>
                <span>Tujuan</span>
              </div>
              <div className="bg-gray-50 rounded-lg p-3 border border-gray-200">
                {destination ? (
                  <div className="text-xs text-gray-600">
                    <div className="font-mono">{destination.latitude.toFixed(6)}, {destination.longitude.toFixed(6)}</div>
                  </div>
                ) : (
                  <div className="text-xs text-gray-400">Belum dipilih</div>
                )}
                <div className="flex gap-2 mt-2">
                  <button
                    onClick={() => {
                      setPickingDestination(!pickingDestination);
                      setPickingOrigin(false);
                    }}
                    className={`flex-1 px-3 py-2 text-xs rounded-md font-medium transition-colors ${
                      pickingDestination
                        ? 'bg-red-500 text-white'
                        : 'bg-white border border-gray-300 text-gray-700 hover:bg-gray-50'
                    }`}
                  >
                    {pickingDestination ? 'Klik peta...' : 'Pilih di peta'}
                  </button>
                  <button
                    onClick={() => getCurrentLocation(false)}
                    className="px-3 py-2 text-xs bg-white border border-gray-300 text-gray-700 rounded-md hover:bg-gray-50 transition-colors font-medium"
                  >
                    Lokasi saya
                  </button>
                </div>
              </div>
            </div>

            {/* Search Button */}
            <button
              onClick={searchRoute}
              disabled={!origin || !destination || loading}
              className="w-full bg-[#FFC107] hover:bg-[#FFA000] disabled:bg-gray-300 disabled:cursor-not-allowed text-[#1a1a1a] font-semibold py-3 px-4 rounded-lg transition-colors shadow-md disabled:shadow-none"
            >
              {loading ? 'Mencari rute...' : 'Cari Rute Termurah'}
            </button>

            {/* Error Message */}
            {error && (
              <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg text-sm">
                {error}
              </div>
            )}

            {/* Route Results */}
            {route && (
              <div className="space-y-4 pb-4">
                {/* Summary */}
                <div className="bg-[#FFC107] rounded-lg p-4 shadow-md">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-semibold text-[#1a1a1a]">Total Biaya</span>
                    <span className="text-xl font-bold text-[#1a1a1a]">
                      {formatCurrency(route.totalFare)}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-semibold text-[#1a1a1a]">Waktu Tempuh</span>
                    <span className="text-sm font-semibold text-[#1a1a1a]">
                      {formatTime(route.totalTimeSeconds)}
                    </span>
                  </div>
                </div>

                {/* Steps */}
                <div className="space-y-3">
                  <h3 className="text-sm font-semibold text-gray-800">Langkah Perjalanan</h3>
                  {route.steps.map((step, index) => (
                    <div
                      key={index}
                      className="bg-white border border-gray-200 rounded-lg p-3 shadow-sm hover:shadow-md transition-shadow"
                    >
                      <div className="flex items-start gap-3">
                        <div className="shrink-0">
                          <div className="w-7 h-7 bg-[#1a1a1a] text-[#FFC107] rounded-full flex items-center justify-center font-bold text-xs shadow-sm">
                            {index + 1}
                          </div>
                        </div>
                        <div className="grow">
                          <p className="font-semibold text-gray-900 text-sm mb-1">
                            {step.fromStopName || step.fromStopId}
                          </p>
                          {step.edgeType === EDGE_TYPE_TRAVEL ? (
                            step.routeId && (
                              <div className="mb-2">
                                <div className="text-xs bg-[#1a1a1a] text-[#FFC107] px-2.5 py-1.5 rounded inline-flex items-center gap-1.5 font-bold shadow-sm">
                                  <span>🚌</span>
                                  <span>Naik bus {step.routeId}</span>
                                </div>
                              </div>
                            )
                          ) : (
                            <div className="mb-2">
                              <div className="text-xs bg-[#FFECB3] text-[#1a1a1a] px-2.5 py-1.5 rounded inline-flex items-center gap-1.5 font-semibold border border-[#FFC107]">
                                <span>🚶</span>
                                <span>
                                  {step.edgeType === 'internal-transfer' ? 'Transfer dalam halte' : 'Transfer ke halte lain'}
                                </span>
                              </div>
                            </div>
                          )}
                          <div className="my-2 flex items-center gap-1">
                            <div className="w-0.5 h-4 bg-gray-300"></div>
                          </div>
                          <p className="font-semibold text-gray-900 text-sm">
                            {step.toStopName || step.toStopId}
                          </p>
                          <div className="mt-2 flex items-center gap-3 text-xs text-gray-600">
                            <span className="flex items-center gap-1">
                              <span>⏱</span>
                              <span>{formatTime(step.timeSeconds)}</span>
                            </span>
                            {step.cost > 0 && (
                              <span className="flex items-center gap-1 text-green-600 font-semibold">
                                <span>💰</span>
                                <span>{formatCurrency(step.cost)}</span>
                              </span>
                            )}
                          </div>
                          {step.edgeType === EDGE_TYPE_TRAVEL && step.availableRoutes && step.availableRoutes.length > 1 && step.routeId && (
                            <div className="mt-2 pt-2 border-t border-gray-100">
                              <p className="text-[9px] text-gray-500 mb-1.5 font-medium">Bus lain tersedia:</p>
                              <div className="flex flex-wrap gap-1">
                                {step.availableRoutes
                                  .filter(route => route !== step.routeId)
                                  .map((route, idx) => (
                                    <span
                                      key={idx}
                                      className="text-[9px] px-2 py-0.5 rounded bg-gray-100 text-gray-700 border border-gray-200"
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
            )}
          </div>
        </div>
      </div>

      {/* Map */}
      <div className="flex-1 h-full">
        <FullScreenMap
          origin={origin}
          destination={destination}
          route={route}
          onMapClick={handleMapClick}
          onOriginDrag={handleOriginDrag}
          onDestinationDrag={handleDestinationDrag}
          pickingOrigin={pickingOrigin}
          pickingDestination={pickingDestination}
        />
      </div>
    </div>
  );
}
