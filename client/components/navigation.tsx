"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { FileText, History } from "lucide-react"
import { cn } from "@/lib/utils"

export function Navigation() {
  const pathname = usePathname()

  return (
    <nav className="border-b border-border bg-card">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center gap-8">
            <Link href="/" className="text-xl font-bold text-foreground">
              ICD-10 Validator
            </Link>
            <div className="flex gap-1">
              <Link
                href="/"
                className={cn(
                  "flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-colors",
                  pathname === "/"
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:text-foreground hover:bg-accent",
                )}
              >
                <FileText className="h-4 w-4" />
                Check Codes
              </Link>
              <Link
                href="/past-cases"
                className={cn(
                  "flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-colors",
                  pathname === "/past-cases"
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:text-foreground hover:bg-accent",
                )}
              >
                <History className="h-4 w-4" />
                Past Cases
              </Link>
            </div>
          </div>
        </div>
      </div>
    </nav>
  )
}
