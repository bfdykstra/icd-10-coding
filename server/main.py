from fastapi import FastAPI
import asyncio
from pydantic import BaseModel

class ICD10Code(BaseModel):
    code: str
    description: str
class CheckRequest(BaseModel):
    discharge_summary: str
    existing_codes: list[ICD10Code]

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post('/check-icd-codes')
async def check_icd_codes(request: CheckRequest):
    ...


@app.post('/check-icd-codes/streaming')
async def check_icd_codes_streaming(request: CheckRequest):
    async def code_checker():
        # Simulate streaming response with asyncio
        async for code in request.existing_codes:
            await asyncio.sleep(1)  # Simulate processing time
            yield {
                "code": code.code,
                "is_relevant": True,  # Placeholder logic
                "confidence": 0.95    # Placeholder confidence score
            }
    return code_checker()