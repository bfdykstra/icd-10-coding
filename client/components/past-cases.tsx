"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { ChevronDown, ChevronUp, Calendar, User } from "lucide-react"
import { useState } from "react"
import { mockPastCases, type PastCase } from "@/lib/mock-past-cases"

export function PastCases() {
  return (
    <div className="container mx-auto px-4 py-8 max-w-6xl">
      <div className="mb-8">
        <h1 className="text-4xl font-bold text-foreground mb-2">Past Cases</h1>
        <p className="text-lg text-muted-foreground">Review previous ICD-10 code validations and outcomes</p>
      </div>

      <div className="space-y-4">
        {mockPastCases.map((pastCase) => (
          <CaseCard key={pastCase.id} pastCase={pastCase} />
        ))}
      </div>
    </div>
  )
}

function CaseCard({ pastCase }: { pastCase: PastCase }) {
  const [isOpen, setIsOpen] = useState(false)

  return (
    <Card className="border-border">
      <Collapsible open={isOpen} onOpenChange={setIsOpen}>
        <CardHeader>
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <CardTitle className="text-foreground flex items-center gap-2">
                Case #{pastCase.id}
                <Badge variant="outline" className="font-normal">
                  {pastCase.acceptedCodes.length} codes accepted
                </Badge>
              </CardTitle>
              <CardDescription className="text-muted-foreground flex items-center gap-4 mt-2">
                <span className="flex items-center gap-1">
                  <Calendar className="h-3 w-3" />
                  {new Date(pastCase.date).toLocaleDateString()}
                </span>
                <span className="flex items-center gap-1">
                  <User className="h-3 w-3" />
                  {pastCase.clinician}
                </span>
              </CardDescription>
            </div>
            <CollapsibleTrigger className="ml-4">
              {isOpen ? (
                <ChevronUp className="h-5 w-5 text-muted-foreground" />
              ) : (
                <ChevronDown className="h-5 w-5 text-muted-foreground" />
              )}
            </CollapsibleTrigger>
          </div>
        </CardHeader>

        <CollapsibleContent>
          <CardContent className="space-y-6">
            {/* Discharge Summary */}
            <div>
              <h3 className="text-sm font-semibold text-foreground mb-2">Discharge Summary</h3>
              <div className="bg-accent rounded-lg p-4">
                <p className="text-sm text-accent-foreground whitespace-pre-wrap">{pastCase.dischargeSummary}</p>
              </div>
            </div>

            {/* Clinician Labeled Codes */}
            <div>
              <h3 className="text-sm font-semibold text-foreground mb-2">Clinician Labeled Codes (Existing)</h3>
              <div className="flex flex-wrap gap-2">
                {pastCase.existingCodes.map((code) => (
                  <Badge key={code.code} variant="secondary" className="bg-secondary text-secondary-foreground">
                    {code.code} - {code.description}
                  </Badge>
                ))}
              </div>
            </div>

            {/* Suggested Missing Codes */}
            <div>
              <h3 className="text-sm font-semibold text-foreground mb-2">Suggested Missing Codes</h3>
              <div className="space-y-2">
                {pastCase.suggestedCodes.map((code) => (
                  <div key={code.code} className="bg-card border border-border rounded-lg p-3">
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <p className="font-medium text-foreground">
                          {code.code} - {code.description}
                        </p>
                        {code.clinicalInfo && <p className="text-sm text-muted-foreground mt-1">{code.clinicalInfo}</p>}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Accepted Codes */}
            <div>
              <h3 className="text-sm font-semibold text-foreground mb-2">Accepted Codes (Final)</h3>
              <div className="flex flex-wrap gap-2">
                {pastCase.acceptedCodes.map((code) => (
                  <Badge key={code.code} variant="default" className="bg-primary text-primary-foreground">
                    {code.code} - {code.description}
                  </Badge>
                ))}
              </div>
            </div>
          </CardContent>
        </CollapsibleContent>
      </Collapsible>
    </Card>
  )
}
