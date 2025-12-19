# TransJakarta Route Finder - Frontend

A modern web application for finding the cheapest and fastest TransJakarta routes based on GPS coordinates.

## Features

- 🗺️ **Interactive Map**: Built with OpenStreetMap and Leaflet for route visualization
- 📍 **Geolocation Support**: Get your current location with one click
- 🎨 **Smooth Animations**: GSAP-powered animations for a polished user experience
- 🚌 **Route Details**: View step-by-step journey information including:
  - Total fare and estimated time
  - Stop names and coordinates
  - Route IDs
  - Walking vs transit steps
- 📱 **Responsive Design**: Works seamlessly on desktop and mobile devices

## Tech Stack

- **Framework**: Next.js 16 with App Router
- **Language**: TypeScript
- **Styling**: Tailwind CSS 4
- **Maps**: Leaflet with React-Leaflet
- **Animations**: GSAP
- **Database ORM**: Drizzle ORM

## Getting Started

### Prerequisites

- Node.js 20+ installed
- Backend API running (see backend documentation)

### Installation

1. Navigate to the frontend directory:
```bash
cd front
```

2. Install dependencies:
```bash
npm install
```

3. Configure environment variables:
```bash
cp .env.example .env
```

Edit `.env` and set:
```env
BACKEND_URL=http://localhost:25200
INTERNAL_API_KEY=your-api-key-here
```

### Development

Run the development server:
```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

### Production Build

Build the application:
```bash
npm run build
```

Start the production server:
```bash
npm start
```

## Usage

1. **Enter Coordinates**:
   - Manually input latitude and longitude for origin and destination
   - Or use the "Gunakan lokasi saat ini" button to auto-fill with your current location

2. **Search Route**:
   - Click "Cari Rute Termurah" to find the cheapest route
   - The backend will calculate the optimal route based on fare

3. **View Results**:
   - See total cost and estimated travel time
   - Review step-by-step directions
   - Visualize the route on the interactive map

## API Integration

The frontend communicates with the backend via `/api/route` endpoint.

### Request Format:
```json
{
  "originLat": -6.2088,
  "originLon": 106.8456,
  "destinationLat": -6.1751,
  "destinationLon": 106.8650
}
```

### Response Format:
```json
{
  "totalFare": 0,
  "totalTimeSeconds": 3721,
  "steps": [
    {
      "fromStopId": "B07312P",
      "fromStopName": "Jln. Pariaman",
      "toStopId": "B06940P",
      "toStopName": "Pasar Rumput 1",
      "edgeType": "travel",
      "cost": 0,
      "timeSeconds": 156,
      "routeId": null,
      "toCoordinates": {
        "latitude": -6.207363,
        "longitude": 106.841524
      }
    }
  ],
  "summary": "..."
}
```

## Project Structure

```
front/
├── src/
│   ├── app/
│   │   ├── components/
│   │   │   ├── RouteSearchForm.tsx  # Search form with geolocation
│   │   │   ├── RouteDisplay.tsx      # Route results display
│   │   │   └── RouteMap.tsx          # Interactive map component
│   │   ├── api/
│   │   │   └── route/
│   │   │       └── route.ts          # API proxy to backend
│   │   ├── layout.tsx                # Root layout
│   │   ├── page.tsx                  # Home page
│   │   └── globals.css               # Global styles
│   └── db/
│       ├── index.ts                  # Database connection
│       └── schema.ts                 # Database schema
├── public/                           # Static assets
├── package.json
└── tsconfig.json
```

## Available Scripts

- `npm run dev` - Start development server with Turbopack
- `npm run build` - Build for production
- `npm start` - Start production server
- `npm run lint` - Run ESLint

## Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## License

This project is part of the TransJakarta Route Finder application.
