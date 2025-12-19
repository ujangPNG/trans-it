#!/bin/bash

set -a
[ -f .env ] && . .env
set +a

echo "INTERNAL_API_KEY: $INTERNAL_API_KEY"
# ... rest of your script

# Test
echo $DATABASE_URL
echo $INTERNAL_API_KEY
API_URL="http://localhost:25200"

echo "=================================="
echo "Testing /route endpoint"
echo "=================================="
echo ""

# Test 1: Using stop IDs/names
echo "Test 1: Route using origin and destination stop IDs/names"
echo "Request:"
cat << 'EOF'
{
  "origin": "STOP_ID_1",
  "destination": "STOP_ID_2"
}
EOF
echo ""
echo "Response:"
curl -X POST "$API_URL/route" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $INTERNAL_API_KEY" \
  -d '{
    "origin": "STOP_ID_1",
    "destination": "STOP_ID_2"
  }' \
  | jq '.'
echo ""
echo "=================================="
echo ""

# Test 2: Using coordinates
echo "Test 2: Route using origin and destination coordinates"
echo "Request:"
cat << 'EOF'
{
  "originLat": -6.2088,
  "originLon": 106.8456,
  "destinationLat": -6.1751,
  "destinationLon": 106.8650
}
EOF
echo ""
echo "Response:"
curl -X POST "$API_URL/route" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $INTERNAL_API_KEY" \
  -d '{
    "originLat": -6.2088,
    "originLon": 106.8456,
    "destinationLat": -6.1751,
    "destinationLon": 106.8650
  }' \
  | jq '.'
echo ""
echo "=================================="
echo ""

# Test 3: Mixed - origin as stop ID, destination as coordinates
echo "Test 3: Mixed - origin as stop ID, destination as coordinates"
echo "Request:"
cat << 'EOF'
{
  "origin": "STOP_ID_1",
  "destinationLat": -6.1751,
  "destinationLon": 106.8650
}
EOF
echo ""
echo "Response:"
curl -X POST "$API_URL/route" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $INTERNAL_API_KEY" \
  -d '{
    "origin": "STOP_ID_1",
    "destinationLat": -6.1751,
    "destinationLon": 106.8650
  }' \
  | jq '.'
echo ""
echo "=================================="
echo ""

# Test 4: Error case - missing required fields
echo "Test 4: Error case - missing both origin and destination"
echo "Request:"
cat << 'EOF'
{}
EOF
echo ""
echo "Response:"
curl -X POST "$API_URL/route" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $INTERNAL_API_KEY" \
  -d '{}' \
  | jq '.'
echo ""
echo "=================================="
echo ""

# Test 5: Health check
echo "Test 5: Health check endpoint"
echo "Response:"
curl -X GET "$API_URL/health" | jq '.'
echo ""
echo "=================================="
echo ""

echo "Testing complete!"
echo ""
echo "Note: Replace STOP_ID_1 and STOP_ID_2 with actual stop IDs from your data."
echo "Note: Coordinates are examples for Jakarta area, adjust as needed."
echo "Note: Install 'jq' for pretty JSON output, or remove '| jq .' to see raw responses."
