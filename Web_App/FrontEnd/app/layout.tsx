export const dynamic = 'force-dynamic';
// export const revalidate: number | false = 0;


import { ThemeProvider } from "@/components/theme-provider";
import type { Metadata } from "next";
import type React from "react";
import "./globals.css";

export const metadata: Metadata = {
  title: "Warehouse Agents Navigation",
  description: "Interactive demonstration of DQN and PPO algorithms in warehouse environments",
    generator: 'v0.dev'
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body suppressHydrationWarning>
        <ThemeProvider attribute="class" defaultTheme="dark">
          {children}
        </ThemeProvider>
      </body>
    </html>
  )
}
