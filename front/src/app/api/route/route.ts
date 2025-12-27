import { NextRequest, NextResponse } from 'next/server';

interface RouteRequestBody {
  origin?: string;
  destination?: string;
  originLat?: number;
  originLon?: number;
  destinationLat?: number;
  destinationLon?: number;
}

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

export async function POST(request: NextRequest) {
  try {
    const body: RouteRequestBody = await request.json();

    // Call backend API with API key
    const backendUrl = process.env.BACKEND_URL || 'http://localhost:25200';
    // Trim quotes from API key in case someone wraps it in quotes
    const apiKey = process.env.INTERNAL_API_KEY?.replace(/^['"]|['"]$/g, '');

    if (!apiKey) {
      return NextResponse.json(
        { error: 'Access denied: API key not configured.' },
        { status: 500 }
      );
    }

    const response = await fetch(`${backendUrl}/route`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': apiKey,
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      let errorMessage = 'Failed to compute route';
      try {
        const text = await response.text();
        try {
          const error = JSON.parse(text);
          errorMessage = error.detail || errorMessage;
        } catch {
          // If response is not JSON, use the text content
          errorMessage = text || errorMessage;
        }
      } catch (e) {
        console.error('Error reading error response:', e);
      }
      
      return NextResponse.json(
        { error: errorMessage },
        { status: response.status }
      );
    }

    const data: RouteResponse = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Route API error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
