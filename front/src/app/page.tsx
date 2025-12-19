'use client';

import { useState, useEffect, useRef } from 'react';
import dynamic from 'next/dynamic';
import RouteSearchForm from './components/RouteSearchForm';
import RouteDisplay from './components/RouteDisplay';
import { gsap } from 'gsap';

// Dynamically import RouteMap to avoid SSR issues with Leaflet
const RouteMap = dynamic(() => import('./components/RouteMap'), {
  ssr: false,
  loading: () => (
    <div className="bg-white rounded-lg shadow-lg h-[500px] flex items-center justify-center">
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
  const [origin, setOrigin] = useState<{ lat: number; lon: number } | undefined>();
  const [destination, setDestination] = useState<{ lat: number; lon: number } | undefined>();
  
  const titleRef = useRef<HTMLHeadingElement>(null);

  useEffect(() => {
    // Animate title on mount
    if (titleRef.current) {
      gsap.fromTo(
        titleRef.current,
        { opacity: 0, y: -50 },
        { opacity: 1, y: 0, duration: 1, ease: 'power3.out' }
      );
    }
  }, []);

  const handleSearch = async (data: {
    originLat: number;
    originLon: number;
    destinationLat: number;
    destinationLon: number;
  }) => {
    setIsLoading(true);
    setError(null);
    setRoute(null);
    setOrigin({ lat: data.originLat, lon: data.originLon });
    setDestination({ lat: data.destinationLat, lon: data.destinationLon });

    try {
      const response = await fetch('/api/route', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          originLat: data.originLat,
          originLon: data.originLon,
          destinationLat: data.destinationLat,
          destinationLon: data.destinationLon,
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

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 py-8 px-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <header className="text-center mb-8">
          <h1
            ref={titleRef}
            className="text-4xl md:text-5xl font-bold text-gray-800 mb-2"
          >
            🚌 TransJakarta Route Finder
          </h1>
          <p className="text-gray-600 text-lg">
            Temukan rute TransJakarta termurah dan tercepat
          </p>
        </header>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left Column - Search Form and Results */}
          <div className="space-y-6">
            <RouteSearchForm onSearch={handleSearch} isLoading={isLoading} />
            {(route || error) && <RouteDisplay route={route} error={error} />}
          </div>

          {/* Right Column - Map */}
          <div className="lg:sticky lg:top-8 h-fit">
            <RouteMap route={route} origin={origin} destination={destination} />
          </div>
        </div>
      </div>
    </div>
  );
}
