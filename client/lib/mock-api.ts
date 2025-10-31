import type { ICD10Code, SuggestedCode } from "@/components/icd10-checker"

type CheckRequest = {
  dischargeSummary: string
  existingCodes: ICD10Code[]
}

// Mock streaming response that simulates AI analysis
export async function mockStreamResponse(
  request: CheckRequest,
  onChunk: (chunk: string) => void,
  onComplete: (codes: SuggestedCode[]) => void,
) {
  const streamingText = `Analyzing discharge summary...

Reviewing existing codes: ${request.existingCodes.map((c) => c.code).join(", ") || "None provided"}

Scanning for missing diagnoses and conditions...

Identifying potential coding gaps...

Analysis complete. Found 3 suggested codes to review.`

  // Simulate streaming by sending chunks
  const words = streamingText.split(" ")
  for (let i = 0; i < words.length; i++) {
    await new Promise((resolve) => setTimeout(resolve, 50))
    onChunk(words[i] + " ")
  }

  // Wait a bit before showing results
  await new Promise((resolve) => setTimeout(resolve, 500))

  // Mock suggested codes
  const suggestedCodes: SuggestedCode[] = [
    {
      code: "E11.65",
      description: "Type 2 diabetes mellitus with hyperglycemia",
      clinicalInfo: "Patient presented with elevated blood glucose levels (320 mg/dL) on admission",
    },
    {
      code: "I50.23",
      description: "Acute on chronic systolic heart failure",
      clinicalInfo: "Documented acute decompensation with reduced ejection fraction (35%)",
    },
    {
      code: "N18.4",
      description: "Chronic kidney disease, stage 4",
      clinicalInfo: "eGFR of 22 mL/min/1.73m² indicates stage 4 CKD, more specific than stage 3",
    },
  ]

  onComplete(suggestedCodes)
}
