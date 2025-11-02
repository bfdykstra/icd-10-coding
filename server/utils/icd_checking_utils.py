#!/usr/bin/env python3
"""
ICD-10 code checking utilities for identifying missing codes in discharge summaries.
"""

from utils.llm_utils import llm_client
import asyncio
from pydantic import BaseModel, Field
from typing import Dict, Any, List, AsyncGenerator, Optional, Union
from utils.vector_store_utils import TypedQueryResult, vector_store, PatientSummaryMetadata
# from utils.embedding_utils import embedding_model
import utils.config as config
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MissingCode(BaseModel):
    """A missing ICD-10 code with clinical evidence"""
    code: str = Field(description="The ICD-10 code")
    description: str = Field(description="The description of the ICD-10 code")
    clinicalInfo: Optional[str] = Field(None, description="Clinical evidence from the discharge summary supporting this code")
    confidence: str = Field(description="Confidence bucket: strong, moderate, weak, unsupported")


class MissingCodesResponse(BaseModel):
    """Response containing list of missing ICD-10 codes"""
    missing_codes: List[MissingCode] = Field(default_factory=list, description="List of missing ICD-10 codes")


missing_code_response_schema = MissingCodesResponse.model_json_schema()


async def check_icd_codes_streaming(discharge_summary: str, existing_codes: list[BaseModel], top_k: int = 20) -> AsyncGenerator[str, None]:
    """
    Check for missing ICD-10 codes against a discharge summary.
    
    Yields SSE-formatted events:
    - progress: Status updates during processing
    - chunk: Partial results as they stream from the LLM
    - result: Final complete result with missing codes
    - done: Completion signal
    - error: Error information if something goes wrong
    
    Args:
        discharge_summary (str): The discharge summary text.
        existing_codes (list[BaseModel]): List of ICD-10 code objects with 'code' and 'description'.
        top_k (int): Number of examples to fetch from the vector database for context.

    Yields:
        str: SSE-formatted event strings
    """
    try:
        # Progress: Searching vector database
        yield _format_sse_event("progress", {"status": "searching", "message": "Querying ICD-10 code database..."})

        # embedding = embedding_model.encode(discharge_summary)
        # examples = vector_store.search_with_embedding(embedding, top_k=top_k)
        examples = {'metadatas': [], 'documents': []}
        # Query vector store for similar examples
        # examples = vector_store.search([discharge_summary], top_k=top_k)
        
        # Progress: Analyzing
        yield _format_sse_event("progress", {"status": "analyzing", "message": "Analyzing discharge summary..."})
        
        # Generate missing codes using LLM with streaming
        # _generate_missing_codes is an async generator function - call it directly
        # Handle streaming chunks
        final_result = None
        async for chunk in _generate_missing_codes(discharge_summary, existing_codes, examples):
            # Yield each chunk as a progress update
            if hasattr(chunk, 'missing_codes'):
                # Chunk is a MissingCodesResponse object
                chunk_data = {
                    "missing_codes": [code.model_dump() if hasattr(code, 'model_dump') else code for code in chunk.missing_codes]
                }
                yield _format_sse_event("chunk", chunk_data)
                final_result = chunk
        
        # Yield final results
        if final_result:
            result_data = {
                "missing_codes": [code.model_dump() for code in final_result.missing_codes]
            }
            yield _format_sse_event("result", result_data)
        else:
            # Fallback if no chunks were received
            yield _format_sse_event("result", {"missing_codes": []})
        
        # Done
        yield _format_sse_event("done", {"status": "complete"})
        
    except Exception as e:
        logger.error(f"Error in check_icd_codes_streaming: {e}", exc_info=True)
        yield _format_sse_event("error", {"status": "error", "message": str(e)})


def _format_sse_event(event_type: str, data: Dict[str, Any]) -> str:
    """Format data as Server-Sent Event"""
    json_data = json.dumps(data)
    return f"event: {event_type}\ndata: {json_data}\n\n"


async def _generate_missing_codes(
    discharge_summary: str, 
    existing_codes: List[BaseModel], 
    examples: TypedQueryResult[PatientSummaryMetadata],
) -> AsyncGenerator[MissingCodesResponse, None]:
    """
    Use instructor to generate structured missing codes response.
    
    Args:
        discharge_summary (str): The discharge summary text.
        existing_codes (list[BaseModel]): List of ICD-10 code objects with 'code' and 'description'.
        examples (TypedQueryResult): Retrieved examples from vector store.
        stream (bool): If True, returns an AsyncGenerator that yields chunks. If False, returns the complete response.

    Returns:
        MissingCodesResponse: Structured response with missing codes when stream=False.
        AsyncGenerator[MissingCodesResponse, None]: Stream of partial MissingCodesResponse objects when stream=True.
    """
    # Get instructor client and raw OpenAI client
    instructor_client, model = llm_client.get_instructor_client()
    raw_client = llm_client.async_client
    
    # Build prompt
    prompt = _build_prompt(discharge_summary, existing_codes, examples)

    # print(prompt)
    
    # Use raw OpenAI client for streaming, then parse JSON with instructor
    # Instructor's streaming with response_model doesn't work as expected
    # So we'll stream raw text and use instructor to parse accumulated JSON
    messages = [
#         {
#             "role": "system",
#             "content": """
#             You are an expert professional ICD-10 medical coder. Your only task is to determine whether the discharge summary supports any additional ICD-10 codes not already recorded.

# Accuracy and documentation integrity are essential.
# If the discharge summary does not clearly support a diagnosis, **do not suggest the code**.
# """
#         },
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    # Stream from raw OpenAI client
    stream = await raw_client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        response_format={"type": "json_object"}  # Request JSON format
    )
    
    # Accumulate the streamed content
    accumulated_content = ""
    async for chunk in stream:
        if chunk.choices and len(chunk.choices) > 0 and chunk.choices[0].delta and chunk.choices[0].delta.content:
            accumulated_content += chunk.choices[0].delta.content
            # Try to parse partial JSON if possible
            # For now, we'll yield the accumulated content as it grows
            # and try to parse it as MissingCodesResponse
            try:
                # Attempt to parse as JSON and create MissingCodesResponse
                parsed = json.loads(accumulated_content)
                partial_response = MissingCodesResponse(**parsed)
                yield partial_response
            except (json.JSONDecodeError, Exception):
                # JSON is incomplete, continue accumulating
                pass
    
    # Final parse of complete response
    try:
        final_parsed = json.loads(accumulated_content)
        final_response = MissingCodesResponse(**final_parsed)
        yield final_response
    except Exception as e:
        logger.error(f"Failed to parse final JSON: {e}, content: {accumulated_content[:200]}")
        # Fallback: try using instructor to parse
        try:
            final_response = await instructor_client.chat.completions.create(
                model=model,
                response_model=MissingCodesResponse,
                messages=messages,
                stream=False,
            )
            yield final_response
        except Exception as fallback_error:
            logger.error(f"Fallback parsing also failed: {fallback_error}")
            # Yield empty response as last resort
            yield MissingCodesResponse(missing_codes=[])


def _build_prompt(discharge_summary: str, existing_codes: List[BaseModel], examples: TypedQueryResult[PatientSummaryMetadata]) -> str:
    """
    Build the prompt for the LLM based on discharge summary and existing codes.
    
    Args:
        discharge_summary (str): The discharge summary text.
        existing_codes (list[BaseModel]): List of ICD-10 code objects with 'code' and 'description'.
        examples (TypedQueryResult): Retrieved examples from vector database.

    Returns:
        str: The constructed prompt string.
    """

    # Existing codes
    existing_codes_str = ""
    if existing_codes:
        for code in existing_codes:
            existing_codes_str += f"- {code.code}: {code.description}\n"
    
    prompt = f"""You are an expert professional ICD-10 medical coder. Your only task is to determine whether the discharge summary supports any additional ICD-10 codes not already recorded:

1. **A clinician’s discharge summary**, and
2. **A list of ICD-10 codes already assigned**

Your goal is to determine **whether any clinically supported ICD-10 diagnoses or conditions are missing**.

**Rules & Requirements**

1. **Only identify codes that are clearly supported by clinical evidence in the discharge summary.**

   * Do *not* assume or infer conditions without documentation.
   * Do *not* code ruled-out, suspected, or irrelevant items unless explicitly documented as confirmed.

2. **Return an empty list if no additional ICD-10 codes are supported.**

3. For each missing code, provide a JSON object with the following fields:

   * **code**: The ICD-10 code
   * **description**: The official ICD-10 description
   * **clinicalInfo**: Clear, specific quotes or data from the discharge summary that justify the code
   * **confidence**: One of: `"strong"`, `"moderate"`, `"weak"`, `"unsupported"`

     * *Use “unsupported” only if there is mention of a condition but insufficient evidence to code it.*

4. **Output must be formatted as a JSON array.**

5. **Do not repeat codes that are already included.**

---

## ✅ **User Input Structure**

**Discharge Summary:**
{discharge_summary}

**Existing ICD-10 Codes:**
{existing_codes_str}
```json
[]
```

Return your analysis as a structured response with the missing codes. The response should be a valid JSON object that matches the following schema:

```json
{missing_code_response_schema}
```

## **TASK**

Input:

1. **Discharge Summary**
   {discharge_summary}

2. **Existing ICD-10 Codes**
   {existing_codes_str}

"""
    return prompt

    
    
    
    
