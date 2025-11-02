"use client";

import { useState } from "react";
import { CheckCircle2, XCircle, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import type { SuggestedCode, ICD10Code } from "./icd10-checker";

type SuggestedCodesProps = {
  codes: SuggestedCode[];
  onAccept: (code: string) => void;
  onReject: (code: string) => void;
  onAddCustom: (code: ICD10Code) => void;
};

export function SuggestedCodes({
  codes,
  onAccept,
  onReject,
  onAddCustom,
}: SuggestedCodesProps) {
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [customCode, setCustomCode] = useState("");
  const [customDescription, setCustomDescription] = useState("");

  const handleAddCustom = () => {
    if (customCode.trim() && customDescription.trim()) {
      onAddCustom({
        code: customCode.trim(),
        description: customDescription.trim(),
      });
      setCustomCode("");
      setCustomDescription("");
      setIsDialogOpen(false);
    }
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-foreground">
          Suggested Missing Codes
        </h3>
        <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
          <DialogTrigger asChild>
            <Button
              variant="outline"
              size="sm"
              className="text-primary border-primary hover:bg-accent bg-transparent"
            >
              <Plus className="h-4 w-4 mr-1" />
              Add Custom
            </Button>
          </DialogTrigger>
          <DialogContent className="bg-card text-card-foreground">
            <DialogHeader>
              <DialogTitle className="text-foreground">
                Add Custom ICD-10 Code
              </DialogTitle>
              <DialogDescription className="text-muted-foreground">
                Enter a code that wasn't suggested but should be included
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <Label htmlFor="code" className="text-foreground">
                  ICD-10 Code
                </Label>
                <Input
                  id="code"
                  placeholder="e.g., I10"
                  value={customCode}
                  onChange={(e) => setCustomCode(e.target.value)}
                  className="bg-card text-card-foreground"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="description" className="text-foreground">
                  Description
                </Label>
                <Input
                  id="description"
                  placeholder="e.g., Essential hypertension"
                  value={customDescription}
                  onChange={(e) => setCustomDescription(e.target.value)}
                  className="bg-card text-card-foreground"
                />
              </div>
            </div>
            <DialogFooter>
              <Button
                onClick={handleAddCustom}
                disabled={!customCode.trim() || !customDescription.trim()}
                className="bg-primary hover:bg-secondary text-primary-foreground"
              >
                Add Code
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      <div className="space-y-2 max-h-[500px] overflow-y-auto">
        {codes.map((code) => (
          <Card
            key={code.code}
            className={`p-4 border-border ${
              code.accepted === true
                ? "bg-accent border-primary"
                : code.accepted === false
                ? "bg-muted opacity-60"
                : "bg-card"
            }`}
          >
            <div className="flex items-start justify-between gap-4">
              <div className="flex-1 space-y-1">
                <div className="flex items-center gap-2 flex-wrap">
                  <Badge
                    variant="outline"
                    className="font-mono text-foreground border-foreground"
                  >
                    {code.code}
                  </Badge>
                  {code.confidence && (
                    <Badge
                      className={
                        code.confidence === "strong"
                          ? "bg-green-600 text-white"
                          : code.confidence === "moderate"
                          ? "bg-yellow-600 text-white"
                          : code.confidence === "weak"
                          ? "bg-orange-600 text-white"
                          : "bg-gray-600 text-white"
                      }
                    >
                      {code.confidence.charAt(0).toUpperCase() +
                        code.confidence.slice(1)}
                    </Badge>
                  )}
                  {code.accepted === true && (
                    <Badge className="bg-primary text-primary-foreground">
                      Accepted
                    </Badge>
                  )}
                  {code.accepted === false && (
                    <Badge
                      variant="secondary"
                      className="bg-muted text-muted-foreground"
                    >
                      Rejected
                    </Badge>
                  )}
                </div>
                <p className="text-sm font-medium text-foreground">
                  {code.description}
                </p>
                {code.clinicalInfo && (
                  <p className="text-sm text-muted-foreground">
                    {code.clinicalInfo}
                  </p>
                )}
              </div>

              {code.accepted === undefined && (
                <div className="flex gap-2">
                  <Button
                    size="sm"
                    onClick={() => onAccept(code.code)}
                    className="bg-primary hover:bg-secondary text-primary-foreground"
                  >
                    <CheckCircle2 className="h-4 w-4" />
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => onReject(code.code)}
                    className="border-border hover:bg-muted"
                  >
                    <XCircle className="h-4 w-4" />
                  </Button>
                </div>
              )}
            </div>
          </Card>
        ))}
      </div>
    </div>
  );
}
