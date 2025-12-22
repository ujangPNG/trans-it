'use client';

import { useState } from 'react';

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
}

interface RouteResponse {
  totalFare: number;
  totalTimeSeconds: number;
  steps: StepResponse[];
  summary: string;
}

interface RouteRequestBody {
  origin?: string;
  destination?: string;
  originLat?: number;
  originLon?: number;
  destinationLat?: number;
  destinationLon?: number;
}

export default function TestPage() {
  const [origin, setOrigin] = useState('');
  const [destination, setDestination] = useState('');
  const [originLat, setOriginLat] = useState('');
  const [originLon, setOriginLon] = useState('');
  const [destLat, setDestLat] = useState('');
  const [destLon, setDestLon] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<RouteResponse | null>(null);
  const [error, setError] = useState('');

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setResult(null);

    try {
      const body: RouteRequestBody = {};
      
      if (origin) body.origin = origin;
      if (destination) body.destination = destination;
      if (originLat) body.originLat = parseFloat(originLat);
      if (originLon) body.originLon = parseFloat(originLon);
      if (destLat) body.destinationLat = parseFloat(destLat);
      if (destLon) body.destinationLon = parseFloat(destLon);

      const response = await fetch('/api/route', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(body),
      });

      const data = await response.json();

      if (!response.ok) {
        setError(data.error || 'Failed to compute route');
        return;
      }

      setResult(data);
    } catch (err) {
      setError('Failed to connect to API');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen p-8 bg-black-100">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold mb-8">Route Testing</h1>

        <form onSubmit={handleSubmit} className="bg-gray-900 p-6 rounded-lg shadow-md mb-8">
          <div className="grid grid-cols-2 gap-6">
            <div>
              <h2 className="text-xl font-semibold mb-4">Origin</h2>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-1">Stop ID/Name</label>
                  <input
                    type="text"
                    value={origin}
                    onChange={(e) => setOrigin(e.target.value)}
                    className="w-full px-3 py-2 border rounded-md"
                    placeholder="e.g., STOP_001"
                  />
                </div>
                <div className="text-center text-sm text-gray-500">OR</div>
                <div>
                  <label className="block text-sm font-medium mb-1">Latitude</label>
                  <input
                    type="number"
                    step="any"
                    value={originLat}
                    onChange={(e) => setOriginLat(e.target.value)}
                    className="w-full px-3 py-2 border rounded-md"
                    placeholder="e.g., -6.2088"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1">Longitude</label>
                  <input
                    type="number"
                    step="any"
                    value={originLon}
                    onChange={(e) => setOriginLon(e.target.value)}
                    className="w-full px-3 py-2 border rounded-md"
                    placeholder="e.g., 106.8456"
                  />
                </div>
              </div>
            </div>

            <div>
              <h2 className="text-xl font-semibold mb-4">Destination</h2>
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-1">Stop ID/Name</label>
                  <input
                    type="text"
                    value={destination}
                    onChange={(e) => setDestination(e.target.value)}
                    className="w-full px-3 py-2 border rounded-md"
                    placeholder="e.g., STOP_002"
                  />
                </div>
                <div className="text-center text-sm text-gray-500">OR</div>
                <div>
                  <label className="block text-sm font-medium mb-1">Latitude</label>
                  <input
                    type="number"
                    step="any"
                    value={destLat}
                    onChange={(e) => setDestLat(e.target.value)}
                    className="w-full px-3 py-2 border rounded-md"
                    placeholder="e.g., -6.1751"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium mb-1">Longitude</label>
                  <input
                    type="number"
                    step="any"
                    value={destLon}
                    onChange={(e) => setDestLon(e.target.value)}
                    className="w-full px-3 py-2 border rounded-md"
                    placeholder="e.g., 106.8650"
                  />
                </div>
              </div>
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="mt-6 w-full bg-blue-600 text-white py-3 rounded-md font-semibold hover:bg-blue-700 disabled:bg-gray-400"
          >
            {loading ? 'Computing Route...' : 'Find Route'}
          </button>
        </form>

        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-8">
            {error}
          </div>
        )}

        {result && (
          <div className="bg-gray-900 p-6 rounded-lg shadow-md">
            <h2 className="text-2xl font-bold mb-4">Route Result</h2>
            
            <div className="mb-6 p-4 bg-gray-800 rounded-lg">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm text-gray-600">Total Fare</p>
                  <p className="text-2xl font-bold text-blue-600">Rp {result.totalFare.toLocaleString()}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Total Time</p>
                  <p className="text-2xl font-bold text-blue-600">{Math.round(result.totalTimeSeconds / 60)} minutes</p>
                </div>
              </div>
              <div className="mt-4">
                <p className="text-sm text-gray-600">Summary</p>
                <p className="text-lg font-medium">{result.summary}</p>
              </div>
            </div>

            <h3 className="text-xl font-semibold mb-4">Steps</h3>
            <div className="space-y-3">
              {result.steps.map((step, idx) => (
                <div key={idx} className="border-l-4 border-blue-500 pl-4 py-2">
                  <div className="flex justify-between items-start">
                    <div>
                      <p className="font-medium">
                        {step.fromStopName || step.fromStopId} → {step.toStopName || step.toStopId}
                      </p>
                      <p className="text-sm text-gray-600">
                        Type: <span className="font-medium">{step.edgeType}</span>
                        {step.routeId && (
                          <span className="ml-2">
                            Route: <span className="font-medium">{step.routeId}</span>
                          </span>
                        )}
                      </p>
                      {step.notes && (
                        <p className="text-sm text-gray-500 italic">{step.notes}</p>
                      )}
                    </div>
                    <div className="text-right">
                      <p className="text-sm font-semibold">Rp {step.cost.toLocaleString()}</p>
                      <p className="text-xs text-gray-600">{Math.round(step.timeSeconds / 60)} min</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
