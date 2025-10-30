"use client"

import type React from "react"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"
import { Upload, FileText } from "lucide-react"
import { ICD10MultiSelect } from "@/components/icd10-multi-select"
import { SuggestedCodes } from "@/components/suggested-codes"
import { mockStreamResponse } from "@/lib/mock-api"

export type ICD10Code = {
  code: string
  description: string
}

export type SuggestedCode = {
  code: string
  description: string
  clinicalInfo?: string
  accepted?: boolean
}

export function ICD10Checker() {
  const [dischargeSummary, setDischargeSummary] = useState("")
  const [selectedCodes, setSelectedCodes] = useState<ICD10Code[]>([])
  const [isProcessing, setIsProcessing] = useState(false)
  const [streamedText, setStreamedText] = useState("")
  const [suggestedCodes, setSuggestedCodes] = useState<SuggestedCode[]>([])
  const [showResults, setShowResults] = useState(false)

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      const reader = new FileReader()
      reader.onload = (event) => {
        setDischargeSummary(event.target?.result as string)
      }
      reader.readAsText(file)
    }
  }

  const handleSubmit = async () => {
    if (!dischargeSummary.trim()) return

    setIsProcessing(true)
    setStreamedText("")
    setSuggestedCodes([])
    setShowResults(true)

    // Mock streaming response
    await mockStreamResponse(
      {
        dischargeSummary,
        existingCodes: selectedCodes,
      },
      (chunk) => {
        setStreamedText((prev) => prev + chunk)
      },
      (codes) => {
        setSuggestedCodes(codes)
        setIsProcessing(false)
      },
    )
  }

  const handleAcceptCode = (code: string) => {
    setSuggestedCodes((prev) => prev.map((c) => (c.code === code ? { ...c, accepted: true } : c)))
  }

  const handleRejectCode = (code: string) => {
    setSuggestedCodes((prev) => prev.map((c) => (c.code === code ? { ...c, accepted: false } : c)))
  }

  const handleAddCustomCode = (newCode: ICD10Code) => {
    setSuggestedCodes((prev) => [
      ...prev,
      {
        code: newCode.code,
        description: newCode.description,
        accepted: true,
      },
    ])
  }

  return (
    <div className="container mx-auto px-4 py-8 max-w-6xl">
      <div className="mb-8 text-center">
        <h1 className="text-4xl font-bold text-foreground mb-2">ICD-10 Code Validator</h1>
        <p className="text-lg text-muted-foreground">
          Ensure accurate coding for proper Medicare & Medicaid reimbursement
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Input Section */}
        <Card className="border-border">
          <CardHeader>
            <CardTitle className="text-foreground">Discharge Summary</CardTitle>
            <CardDescription className="text-muted-foreground">
              Paste or upload the patient's discharge summary
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <Label htmlFor="summary" className="text-foreground">
                Summary Text
              </Label>
              <Textarea
                id="summary"
                placeholder="Paste discharge summary here..."
                value={dischargeSummary}
                onChange={(e) => setDischargeSummary(e.target.value)}
                className="min-h-[200px] mt-2 bg-card text-card-foreground"
              />
            </div>

            <div className="flex items-center gap-2">
              <div className="h-px flex-1 bg-border" />
              <span className="text-sm text-muted-foreground">or</span>
              <div className="h-px flex-1 bg-border" />
            </div>

            <div>
              <Label
                htmlFor="file-upload"
                className="flex items-center justify-center gap-2 cursor-pointer border-2 border-dashed border-border rounded-lg p-6 hover:border-primary transition-colors bg-card"
              >
                <Upload className="h-5 w-5 text-muted-foreground" />
                <span className="text-sm text-muted-foreground">Upload discharge summary file</span>
                <input
                  id="file-upload"
                  type="file"
                  accept=".txt,.doc,.docx"
                  onChange={handleFileUpload}
                  className="sr-only"
                />
              </Label>
            </div>

            <div>
              <Label className="text-foreground mb-2 block">Existing ICD-10 Codes</Label>
              <ICD10MultiSelect selectedCodes={selectedCodes} onCodesChange={setSelectedCodes} />
            </div>

            <Button
              onClick={handleSubmit}
              disabled={!dischargeSummary.trim() || isProcessing}
              className="w-full bg-primary hover:bg-secondary text-primary-foreground"
              size="lg"
            >
              {isProcessing ? (
                <>
                  <div className="h-4 w-4 border-2 border-primary-foreground border-t-transparent rounded-full animate-spin mr-2" />
                  Processing...
                </>
              ) : (
                <>
                  <FileText className="h-4 w-4 mr-2" />
                  Submit for Checking
                </>
              )}
            </Button>
          </CardContent>
        </Card>

        {/* Results Section */}
        <Card className="border-border">
          <CardHeader>
            <CardTitle className="text-foreground">Analysis Results</CardTitle>
            <CardDescription className="text-muted-foreground">
              {showResults
                ? "Review suggested codes and accept or reject them"
                : "Results will appear here after submission"}
            </CardDescription>
          </CardHeader>
          <CardContent>
            {!showResults ? (
              <div className="flex items-center justify-center h-[400px] text-muted-foreground">
                <div className="text-center">
                  <FileText className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>Submit a discharge summary to begin analysis</p>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                {isProcessing && (
                  <div className="bg-accent rounded-lg p-4">
                    <p className="text-sm text-accent-foreground whitespace-pre-wrap">{streamedText}</p>
                  </div>
                )}

                {!isProcessing && suggestedCodes.length > 0 && (
                  <SuggestedCodes
                    codes={suggestedCodes}
                    onAccept={handleAcceptCode}
                    onReject={handleRejectCode}
                    onAddCustom={handleAddCustomCode}
                  />
                )}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Summary of Accepted Codes */}
      {showResults && !isProcessing && suggestedCodes.length > 0 && (
        <Card className="mt-6 border-border">
          <CardHeader>
            <CardTitle className="text-foreground">Accepted Codes Summary</CardTitle>
            <CardDescription className="text-muted-foreground">
              Codes marked for inclusion in the final report
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              {suggestedCodes
                .filter((code) => code.accepted)
                .map((code) => (
                  <Badge key={code.code} variant="default" className="bg-primary text-primary-foreground">
                    {code.code} - {code.description}
                  </Badge>
                ))}
              {suggestedCodes.filter((code) => code.accepted).length === 0 && (
                <p className="text-sm text-muted-foreground">No codes accepted yet</p>
              )}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
