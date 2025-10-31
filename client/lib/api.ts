export type ICD10Code = {
  code: string
  description: string
}

export type SuggestedCode = {
  code: string
  description: string
  clinicalInfo?: string
  confidence?: string
  accepted?: boolean
}

type ProgressEvent = {
  status: string
  message: string
}

type ChunkEvent = {
  missing_codes: Array<{
    code: string
    description: string
    clinicalInfo?: string
    confidence?: string
  }>
}

type ResultEvent = {
  missing_codes: Array<{
    code: string
    description: string
    clinicalInfo?: string
    confidence?: string
  }>
}

type ErrorEvent = {
  status: string
  message: string
}

/**
 * Connect to the streaming API endpoint and process ICD-10 code suggestions
 */
export async function checkIcdCodesStreaming(
  dischargeSummary: string,
  existingCodes: ICD10Code[],
  onProgress: (message: string) => void,
  onChunk: (codes: SuggestedCode[]) => void,
  onComplete: (codes: SuggestedCode[]) => void,
  onError: (error: string) => void
): Promise<void> {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
  
  try {
    const response = await fetch(`${apiUrl}/check-icd-codes/streaming`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        discharge_summary: dischargeSummary,
        existing_codes: existingCodes,
      }),
    })

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`)
    }

    if (!response.body) {
      throw new Error('Response body is null')
    }

    // Process the SSE stream
    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      
      if (done) {
        break
      }

      // Decode the chunk and add to buffer
      buffer += decoder.decode(value, { stream: true })

      // Process complete SSE messages
      const lines = buffer.split('\n\n')
      buffer = lines.pop() || '' // Keep incomplete message in buffer

      for (const line of lines) {
        if (!line.trim()) continue

        try {
          // Parse SSE format: "event: type\ndata: json"
          const eventMatch = line.match(/event:\s*(\w+)\ndata:\s*(.+)/s)
          
          if (!eventMatch) continue

          const [, eventType, eventData] = eventMatch
          const data = JSON.parse(eventData)

          switch (eventType) {
            case 'progress':
              const progressData = data as ProgressEvent
              onProgress(progressData.message)
              break

            case 'chunk':
              const chunkData = data as ChunkEvent
              if (chunkData.missing_codes && chunkData.missing_codes.length > 0) {
                const suggestedCodes: SuggestedCode[] = chunkData.missing_codes.map(code => ({
                  code: code.code,
                  description: code.description,
                  clinicalInfo: code.clinicalInfo,
                  confidence: code.confidence,
                }))
                onChunk(suggestedCodes)
              }
              break

            case 'result':
              const resultData = data as ResultEvent
              const finalCodes: SuggestedCode[] = resultData.missing_codes.map(code => ({
                code: code.code,
                description: code.description,
                clinicalInfo: code.clinicalInfo,
                confidence: code.confidence,
              }))
              onComplete(finalCodes)
              break

            case 'done':
              // Stream completed successfully
              return

            case 'error':
              const errorData = data as ErrorEvent
              onError(errorData.message || 'An error occurred during processing')
              return

            default:
              console.warn(`Unknown event type: ${eventType}`)
          }
        } catch (parseError) {
          console.error('Error parsing SSE message:', parseError, line)
        }
      }
    }
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Failed to connect to API'
    onError(errorMessage)
  }
}

