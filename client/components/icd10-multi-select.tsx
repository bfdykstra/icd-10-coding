"use client"

import { useState } from "react"
import { Check, ChevronsUpDown, X } from "lucide-react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from "@/components/ui/command"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { Badge } from "@/components/ui/badge"
import type { ICD10Code } from "./icd10-checker"

// Sample ICD-10 codes for demonstration
const SAMPLE_ICD10_CODES: ICD10Code[] = [
  { code: "I10", description: "Essential (primary) hypertension" },
  { code: "E11.9", description: "Type 2 diabetes mellitus without complications" },
  { code: "J44.9", description: "Chronic obstructive pulmonary disease, unspecified" },
  { code: "I25.10", description: "Atherosclerotic heart disease of native coronary artery" },
  { code: "N18.3", description: "Chronic kidney disease, stage 3" },
  { code: "I50.9", description: "Heart failure, unspecified" },
  { code: "E78.5", description: "Hyperlipidemia, unspecified" },
  { code: "F17.210", description: "Nicotine dependence, cigarettes, uncomplicated" },
  { code: "Z79.4", description: "Long term (current) use of insulin" },
  { code: "I48.91", description: "Unspecified atrial fibrillation" },
]

type ICD10MultiSelectProps = {
  selectedCodes: ICD10Code[]
  onCodesChange: (codes: ICD10Code[]) => void
}

export function ICD10MultiSelect({ selectedCodes, onCodesChange }: ICD10MultiSelectProps) {
  const [open, setOpen] = useState(false)

  const handleSelect = (code: ICD10Code) => {
    const isSelected = selectedCodes.some((c) => c.code === code.code)
    if (isSelected) {
      onCodesChange(selectedCodes.filter((c) => c.code !== code.code))
    } else {
      onCodesChange([...selectedCodes, code])
    }
  }

  const handleRemove = (code: string) => {
    onCodesChange(selectedCodes.filter((c) => c.code !== code))
  }

  return (
    <div className="space-y-2">
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button
            variant="outline"
            role="combobox"
            aria-expanded={open}
            className="w-full justify-between bg-card text-card-foreground hover:bg-accent hover:text-accent-foreground"
          >
            <span className="text-muted-foreground">
              {selectedCodes.length > 0
                ? `${selectedCodes.length} code${selectedCodes.length > 1 ? "s" : ""} selected`
                : "Select existing codes..."}
            </span>
            <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-full p-0 bg-popover" align="start">
          <Command className="bg-popover">
            <CommandInput placeholder="Search ICD-10 codes..." className="text-popover-foreground" />
            <CommandList>
              <CommandEmpty className="text-muted-foreground">No codes found.</CommandEmpty>
              <CommandGroup>
                {SAMPLE_ICD10_CODES.map((code) => {
                  const isSelected = selectedCodes.some((c) => c.code === code.code)
                  return (
                    <CommandItem
                      key={code.code}
                      value={`${code.code} ${code.description}`}
                      onSelect={() => handleSelect(code)}
                      className="text-popover-foreground"
                    >
                      <Check className={cn("mr-2 h-4 w-4", isSelected ? "opacity-100" : "opacity-0")} />
                      <div className="flex flex-col">
                        <span className="font-medium">{code.code}</span>
                        <span className="text-sm text-muted-foreground">{code.description}</span>
                      </div>
                    </CommandItem>
                  )
                })}
              </CommandGroup>
            </CommandList>
          </Command>
        </PopoverContent>
      </Popover>

      {selectedCodes.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {selectedCodes.map((code) => (
            <Badge key={code.code} variant="secondary" className="bg-secondary text-secondary-foreground">
              {code.code}
              <button onClick={() => handleRemove(code.code)} className="ml-2 hover:text-destructive">
                <X className="h-3 w-3" />
              </button>
            </Badge>
          ))}
        </div>
      )}
    </div>
  )
}
