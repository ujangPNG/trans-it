'use client';

import { useState, useEffect, useRef } from 'react';
import { gsap } from 'gsap';

interface RouteSearchFormProps {
  onSearch: (data: {
    originLat: number;
    originLon: number;
    destinationLat: number;
    destinationLon: number;
  }) => void;
  isLoading: boolean;
}

export default function RouteSearchForm({ onSearch, isLoading }: RouteSearchFormProps) {
  const [originLat, setOriginLat] = useState('');
  const [originLon, setOriginLon] = useState('');
  const [destinationLat, setDestinationLat] = useState('');
  const [destinationLon, setDestinationLon] = useState('');
  const [locationError, setLocationError] = useState('');

  const formRef = useRef<HTMLDivElement>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    // GSAP entrance animation
    if (formRef.current) {
      gsap.fromTo(
        formRef.current,
        { opacity: 0, y: -30 },
        { opacity: 1, y: 0, duration: 0.8, ease: 'power3.out' }
      );
    }
  }, []);

  const getCurrentLocation = (isOrigin: boolean) => {
    if (!navigator.geolocation) {
      setLocationError('Geolocation is not supported by your browser');
      return;
    }

    setLocationError('');
    navigator.geolocation.getCurrentPosition(
      (position) => {
        if (isOrigin) {
          setOriginLat(position.coords.latitude.toFixed(6));
          setOriginLon(position.coords.longitude.toFixed(6));
        } else {
          setDestinationLat(position.coords.latitude.toFixed(6));
          setDestinationLon(position.coords.longitude.toFixed(6));
        }
        
        // Animate the input fields
        const inputs = document.querySelectorAll(isOrigin ? '.origin-input' : '.destination-input');
        gsap.fromTo(
          inputs,
          { scale: 1.05, backgroundColor: '#dcfce7' },
          { scale: 1, backgroundColor: '#ffffff', duration: 0.5 }
        );
      },
      (error) => {
        setLocationError('Unable to retrieve your location: ' + error.message);
      }
    );
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    const parsedOriginLat = parseFloat(originLat);
    const parsedOriginLon = parseFloat(originLon);
    const parsedDestinationLat = parseFloat(destinationLat);
    const parsedDestinationLon = parseFloat(destinationLon);

    if (
      isNaN(parsedOriginLat) ||
      isNaN(parsedOriginLon) ||
      isNaN(parsedDestinationLat) ||
      isNaN(parsedDestinationLon)
    ) {
      setLocationError('Please enter valid coordinates');
      return;
    }

    // Animate button on submit
    if (buttonRef.current) {
      gsap.to(buttonRef.current, {
        scale: 0.95,
        duration: 0.1,
        yoyo: true,
        repeat: 1,
      });
    }

    onSearch({
      originLat: parsedOriginLat,
      originLon: parsedOriginLon,
      destinationLat: parsedDestinationLat,
      destinationLon: parsedDestinationLon,
    });
  };

  return (
    <div ref={formRef} className="bg-white rounded-lg shadow-lg p-6 mb-6">
      <h2 className="text-2xl font-bold mb-4 text-gray-800">Cari Rute TransJakarta</h2>
      
      {locationError && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
          {locationError}
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Origin */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Titik Awal
          </label>
          <div className="grid grid-cols-2 gap-2 mb-2">
            <input
              type="text"
              placeholder="Latitude"
              value={originLat}
              onChange={(e) => setOriginLat(e.target.value)}
              className="origin-input px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent text-gray-800"
              required
            />
            <input
              type="text"
              placeholder="Longitude"
              value={originLon}
              onChange={(e) => setOriginLon(e.target.value)}
              className="origin-input px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent text-gray-800"
              required
            />
          </div>
          <button
            type="button"
            onClick={() => getCurrentLocation(true)}
            className="text-sm text-blue-600 hover:text-blue-800 font-medium"
          >
            📍 Gunakan lokasi saat ini
          </button>
        </div>

        {/* Destination */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Tujuan
          </label>
          <div className="grid grid-cols-2 gap-2 mb-2">
            <input
              type="text"
              placeholder="Latitude"
              value={destinationLat}
              onChange={(e) => setDestinationLat(e.target.value)}
              className="destination-input px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent text-gray-800"
              required
            />
            <input
              type="text"
              placeholder="Longitude"
              value={destinationLon}
              onChange={(e) => setDestinationLon(e.target.value)}
              className="destination-input px-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent text-gray-800"
              required
            />
          </div>
          <button
            type="button"
            onClick={() => getCurrentLocation(false)}
            className="text-sm text-blue-600 hover:text-blue-800 font-medium"
          >
            📍 Gunakan lokasi saat ini
          </button>
        </div>

        <button
          ref={buttonRef}
          type="submit"
          disabled={isLoading}
          className={`w-full py-3 px-4 rounded-md font-semibold text-white transition-colors ${
            isLoading
              ? 'bg-gray-400 cursor-not-allowed'
              : 'bg-blue-600 hover:bg-blue-700'
          }`}
        >
          {isLoading ? 'Mencari rute...' : 'Cari Rute Termurah'}
        </button>
      </form>
    </div>
  );
}
