import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "TransJakarta Route Finder - Temukan Rute Termurah",
  description: "Aplikasi untuk mencari rute TransJakarta termurah dan tercepat berdasarkan koordinat lokasi Anda",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className="antialiased">
        {children}
      </body>
    </html>
  );
}
