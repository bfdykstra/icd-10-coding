"use client";

import { useState, useMemo, memo, useCallback } from "react";
import { Check, ChevronsUpDown, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Badge } from "@/components/ui/badge";
import type { ICD10Code } from "./icd10-checker";
import ICD10_CODES_JSON from "@/assets/icd10Data/icd10_codes.json";

type ICD10MultiSelectProps = {
  selectedCodes: ICD10Code[];
  onCodesChange: (codes: ICD10Code[]) => void;
};

// Limit results for performance - show max 100 items at a time
const MAX_DISPLAYED_CODES = 100;

// Memoized component to prevent unnecessary re-renders
export const ICD10MultiSelect = memo(function ICD10MultiSelect({
  selectedCodes,
  onCodesChange,
}: ICD10MultiSelectProps) {
  const [open, setOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");

  // Clear search when popover closes
  const handleOpenChange = useCallback((isOpen: boolean) => {
    setOpen(isOpen);
    if (!isOpen) {
      setSearchQuery("");
    }
  }, []);

  // Memoize selected codes set for O(1) lookup instead of O(n)
  const selectedCodesSet = useMemo(
    () => new Set(selectedCodes.map((c) => c.code)),
    [selectedCodes]
  );

  const handleSelect = useCallback(
    (code: ICD10Code) => {
      const isSelected = selectedCodesSet.has(code.code);
      if (isSelected) {
        onCodesChange(selectedCodes.filter((c) => c.code !== code.code));
      } else {
        onCodesChange([...selectedCodes, code]);
      }
    },
    [selectedCodesSet, selectedCodes, onCodesChange]
  );

  const handleRemove = useCallback(
    (code: string) => {
      onCodesChange(selectedCodes.filter((c) => c.code !== code));
    },
    [selectedCodes, onCodesChange]
  );

  // Filter and limit codes - only compute when popover is open
  const filteredCodes = useMemo(() => {
    if (!open) return [];

    let filtered = ICD10_CODES_JSON;

    // If there's a search query, filter the codes
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      filtered = ICD10_CODES_JSON.filter(
        (code) =>
          code.code.toLowerCase().includes(query) ||
          code.description.toLowerCase().includes(query)
      );
    }

    // Limit to MAX_DISPLAYED_CODES for performance
    return filtered.slice(0, MAX_DISPLAYED_CODES);
  }, [open, searchQuery]);

  const hasMoreResults = useMemo(() => {
    if (!searchQuery.trim()) {
      return ICD10_CODES_JSON.length > MAX_DISPLAYED_CODES;
    }
    const query = searchQuery.toLowerCase();
    const totalMatches = ICD10_CODES_JSON.filter(
      (code) =>
        code.code.toLowerCase().includes(query) ||
        code.description.toLowerCase().includes(query)
    ).length;
    return totalMatches > MAX_DISPLAYED_CODES;
  }, [searchQuery]);

  // Memoize the command items - only for visible filtered codes
  const commandItems = useMemo(
    () =>
      filteredCodes.map((code) => {
        const isSelected = selectedCodesSet.has(code.code);
        return (
          <CommandItem
            key={code.code}
            value={`${code.code} ${code.description}`}
            onSelect={() => handleSelect(code)}
            className="text-popover-foreground"
          >
            <Check
              className={cn(
                "mr-2 h-4 w-4 shrink-0",
                isSelected ? "opacity-100" : "opacity-0"
              )}
            />
            <div className="flex flex-col min-w-0 flex-1">
              <span className="font-medium">{code.code}</span>
              <span className="text-sm text-muted-foreground whitespace-normal wrap-break-word">
                {code.description}
              </span>
            </div>
          </CommandItem>
        );
      }),
    [filteredCodes, selectedCodesSet, handleSelect]
  );

  return (
    <div className="space-y-2">
      <Popover open={open} onOpenChange={handleOpenChange}>
        <PopoverTrigger asChild>
          <Button
            variant="outline"
            role="combobox"
            aria-expanded={open}
            className="w-full justify-between bg-card text-card-foreground hover:bg-accent hover:text-accent-foreground"
          >
            <span className="text-muted-foreground">
              {selectedCodes.length > 0
                ? `${selectedCodes.length} code${
                    selectedCodes.length > 1 ? "s" : ""
                  } selected`
                : "Select existing codes..."}
            </span>
            <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-[500px] p-0 bg-popover" align="start">
          <Command className="bg-popover" shouldFilter={false}>
            <CommandInput
              placeholder="Search ICD-10 codes..."
              className="text-popover-foreground"
              value={searchQuery}
              onValueChange={setSearchQuery}
            />
            <CommandList>
              <CommandEmpty className="text-muted-foreground">
                No codes found.
              </CommandEmpty>
              <CommandGroup>
                {commandItems}
                {hasMoreResults && (
                  <div className="px-2 py-1.5 text-xs text-muted-foreground text-center border-t">
                    Showing first {MAX_DISPLAYED_CODES} results. Keep typing to
                    narrow down...
                  </div>
                )}
              </CommandGroup>
            </CommandList>
          </Command>
        </PopoverContent>
      </Popover>

      {selectedCodes.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {selectedCodes.map((code) => (
            <Badge
              key={code.code}
              variant="secondary"
              className="bg-secondary text-secondary-foreground"
            >
              {code.code}
              <button
                onClick={() => handleRemove(code.code)}
                className="ml-2 hover:text-destructive"
              >
                <X className="h-3 w-3" />
              </button>
            </Badge>
          ))}
        </div>
      )}
    </div>
  );
});
