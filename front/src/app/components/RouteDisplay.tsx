'use client';

import { useEffect, useRef } from 'react';
import { gsap } from 'gsap';

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

interface RouteDisplayProps {
  route: RouteResponse | null;
  error: string | null;
}

export default function RouteDisplay({ route, error }: RouteDisplayProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const stepsRef = useRef<HTMLDivElement[]>([]);

  useEffect(() => {
    if (route && containerRef.current) {
      // Animate container entrance
      gsap.fromTo(
        containerRef.current,
        { opacity: 0, x: 50 },
        { opacity: 1, x: 0, duration: 0.6, ease: 'power3.out' }
      );

      // Stagger animation for steps
      if (stepsRef.current.length > 0) {
        gsap.fromTo(
          stepsRef.current,
          { opacity: 0, y: 20 },
          {
            opacity: 1,
            y: 0,
            duration: 0.4,
            stagger: 0.1,
            ease: 'power2.out',
          }
        );
      }
    }
  }, [route]);

  useEffect(() => {
    if (error && containerRef.current) {
      gsap.fromTo(
        containerRef.current,
        { opacity: 0, scale: 0.9 },
        { opacity: 1, scale: 1, duration: 0.3, ease: 'back.out(1.7)' }
      );
    }
  }, [error]);

  if (error) {
    return (
      <div ref={containerRef} className="bg-red-50 border border-red-300 rounded-lg p-6">
        <h3 className="text-xl font-bold text-red-800 mb-2">Error</h3>
        <p className="text-red-700">{error}</p>
      </div>
    );
  }

  if (!route) {
    return null;
  }

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

  return (
    <div ref={containerRef} className="bg-white rounded-lg shadow-lg p-6">
      <h2 className="text-2xl font-bold mb-4 text-gray-800">Hasil Rute</h2>

      {/* Summary */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="text-sm text-gray-600">Total Biaya</p>
            <p className="text-2xl font-bold text-blue-700">
              {formatCurrency(route.totalFare)}
            </p>
          </div>
          <div>
            <p className="text-sm text-gray-600">Estimasi Waktu</p>
            <p className="text-2xl font-bold text-blue-700">
              {formatTime(route.totalTimeSeconds)}
            </p>
          </div>
        </div>
      </div>

      {/* Steps */}
      <div className="space-y-3">
        <h3 className="text-lg font-semibold text-gray-700 mb-3">Langkah Perjalanan</h3>
        {route.steps.map((step, index) => (
          <div
            key={index}
            ref={(el) => {
              if (el) stepsRef.current[index] = el;
            }}
            className="bg-gray-50 border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow"
          >
            <div className="flex items-start gap-3">
              <div className="flex-shrink-0">
                <div className="w-8 h-8 bg-blue-600 text-white rounded-full flex items-center justify-center font-bold text-sm">
                  {index + 1}
                </div>
              </div>
              <div className="flex-grow">
                <div className="flex justify-between items-start mb-2">
                  <div>
                    <p className="font-semibold text-gray-800">
                      {step.fromStopName || step.fromStopId}
                    </p>
                    <p className="text-sm text-gray-500">↓</p>
                    <p className="font-semibold text-gray-800">
                      {step.toStopName || step.toStopId}
                    </p>
                  </div>
                  <div className="text-right">
                    <p className="text-sm text-gray-600">
                      {formatTime(step.timeSeconds)}
                    </p>
                    {step.cost > 0 && (
                      <p className="text-sm font-semibold text-green-600">
                        {formatCurrency(step.cost)}
                      </p>
                    )}
                  </div>
                </div>
                {step.routeId && (
                  <div className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded inline-block">
                    {step.routeId}
                  </div>
                )}
                {step.notes && (
                  <p className="text-xs text-gray-500 mt-2">{step.notes}</p>
                )}
                {step.edgeType && (
                  <p className="text-xs text-gray-400 mt-1">
                    Tipe: {step.edgeType}
                  </p>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
