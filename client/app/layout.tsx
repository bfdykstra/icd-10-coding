import type React from "react";
import type { Metadata } from "next";

import { Analytics } from "@vercel/analytics/next";
import "./globals.css";

import {
  Noto_Sans as V0_Font_Noto_Sans,
  Roboto_Mono as V0_Font_Roboto_Mono,
  Source_Serif_4 as V0_Font_Source_Serif_4,
} from "next/font/google";
import { Navigation } from "@/components/navigation";

// Initialize fonts
const _notoSans = V0_Font_Noto_Sans({
  subsets: ["latin"],
  weight: ["100", "200", "300", "400", "500", "600", "700", "800", "900"],
});
const _robotoMono = V0_Font_Roboto_Mono({
  subsets: ["latin"],
  weight: ["100", "200", "300", "400", "500", "600", "700"],
});
const _sourceSerif_4 = V0_Font_Source_Serif_4({
  subsets: ["latin"],
  weight: ["200", "300", "400", "500", "600", "700", "800", "900"],
});

export const metadata: Metadata = {
  title: "ICD-10 Coding Assistant",
  description: "Find missing ICD-10 codes",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`font-sans antialiased`}>
        <Navigation />
        {children}
        <Analytics />
      </body>
    </html>
  );
}
