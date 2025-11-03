import os
# Set tokenizers parallelism to avoid warnings when using uvicorn with reload
# This must be set before any tokenizers are imported/initialized
# os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from utils.icd_checking_utils import check_icd_codes_streaming

class ICD10Code(BaseModel):
    code: str
    description: str

class CheckRequest(BaseModel):
    discharge_summary: str
    existing_codes: list[ICD10Code]

app = FastAPI()

# Configure CORS to allow requests from Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://icd-10-coding.vercel.app",  # Removed trailing slash - CORS matching is exact
        "http://localhost:3000",  # Next.js default dev server
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers
)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post('/check-icd-codes')
async def check_icd_codes(request: CheckRequest):
    ...


@app.post('/check-icd-codes/streaming')
async def check_icd_codes_streaming_endpoint(request: CheckRequest):
    """
    Stream missing ICD-10 codes as Server-Sent Events (SSE).
    
    Returns SSE-formatted events:
    - progress: Status updates during processing
    - result: Final missing codes result
    - done: Completion signal
    - error: Error information if something goes wrong
    """
    # Get the async generator from the utility function
    # ICD10Code models are BaseModel instances, so they work directly
    async def event_generator():
        async for event in check_icd_codes_streaming(
            discharge_summary=request.discharge_summary,
            existing_codes=request.existing_codes,  # These are BaseModel instances
            top_k = 5
        ):
            yield event
    
    # Return StreamingResponse with SSE media type
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable buffering for nginx
        }
    )