import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

// middleware anti bot

const blockedBots = [
  // 'Googlebot',
  // 'Google-InspectionTool',
  // 'Google-Structured-Data-Testing-Tool',
  // 'Bingbot',
  // 'Curl',
  // 'curl',
  'AhrefsBot',
  'SemrushBot',
  'MJ12bot',
  'DotBot',
  'YandexBot',
  'facebookexternalhit',
  'Facebot',
  'GPTBot',
  // 'Pinterestbot',
  'Twitterbot',
  'PetalBot',
  'Sogou',
  'Baiduspider',
  'CriteoBot',
  'DuckDuckBot',
  'Slurp',
  'SeznamBot'
];

export function proxy(request: NextRequest) {
  const ua = request.headers.get('user-agent') || '';

  const forwardedFor = request.headers.get('x-forwarded-for');
  const realIp = request.headers.get('x-real-ip');

  const ip =
    forwardedFor?.split(',')[0]?.trim() ||
    realIp ||
    'unknown';

  const isBlocked = blockedBots.some(bot => ua.includes(bot));
  // console.log(`[UA CHECK] UA: "${ua}" | IP: ${ip} | Path: ${request.nextUrl.pathname}`); //aish ini bising kat console

  if (isBlocked) {
  fetch(`${request.nextUrl.origin}/api/log`, {
    method: 'GET',
    body: JSON.stringify({
      message: `[BLOCKED BOT] UA: "${ua}" | IP: ${ip} | Path: ${request.nextUrl.pathname}`
    }),
    headers: { 'Content-Type': 'application/json' }
  }).catch(() => {});
  return new NextResponse('Blocked by middleware, this app is still in development.', { status: 403 });
}

}

export const config = {
  matcher: '/:path*',
};
// thanks kodenya bing copilot :D