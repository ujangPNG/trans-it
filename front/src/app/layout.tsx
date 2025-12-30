import type { Metadata } from "next";
import "./globals.css";
import { GoogleAnalytics } from '@next/third-parties/google'

export const metadata: Metadata = {
  title: "TransIt - Temukan Rute Termurah",
  description: "WebApp untuk mencari rute TransJakarta termurah dan tercepat berdasarkan koordinat lokasi Anda",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <GoogleAnalytics gaId="G-WG488C5JB4" />
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}
