
from llm_utils import llm_client
import asyncio
from pydantic import BaseModel
from typing import Dict, Any, List, AsyncGenerator
from vector_store_utils import TypedQueryResult, vector_store, TypedQueryResult
import config



import logging
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



async def check_icd_codes_streaming(discharge_summary: str, existing_codes: list[BaseModel], top_k = 20) -> AsyncGenerator[Dict[str, Any]]:
    """
    Check relevance of existing ICD-10 codes against a discharge summary.

    Yields real-time events during the checking process:
    - progress updates
    - intermediate results
    - synthesized response chunks
    
    Args:
        discharge_summary (str): The discharge summary text.
        existing_codes (list[BaseModel]): List of ICD-10 code objects with 'code' and 'description'.
        top_k (int): Number of examples to fetch from the vector database for context.

    """
    ...



async def _generate_icd_codes_streaming(discharge_summary: str, existing_codes: List[BaseModel], top_k = 20) -> AsyncGenerator[str, None]:
    """
    Async generator to stream relevance checking of ICD-10 codes.

    Given the discharge summary and the existing ICD-10 codes
    query the vector db for relevant examples using the discharge summary as the query document,
    build the prompt that checks for missing codes
    and stream the LLM's response back as JSON strings.
    
    Args:
        discharge_summary (str): The discharge summary text.
        existing_codes (list[BaseModel]): List of ICD-10 code objects with 'code' and 'description'.
        top_k (int): Number of examples to fetch from the vector database for context.

    Yields:
        str: Streaming JSON strings with code relevance results.
    """
    examples = vector_store.search([discharge_summary], top_k=top_k)

    #get the full icd-10 codes from the returned examples
    prompt = await _build_prompt(discharge_summary, existing_codes, examples)




async def _build_prompt(discharge_summary: str, existing_codes: List[BaseModel], examples: TypedQueryResult) -> str:
    """
    Build the prompt for the LLM based on discharge summary and existing codes.
    
    Args:
        discharge_summary (str): The discharge summary text.
        existing_codes (list[BaseModel]): List of ICD-10 code objects with 'code' and 'description'.
        examples (list[Dict[str, Any]]): List of example contexts from the vector database.

    Returns:
        str: The constructed prompt string.
    """
    ...
    
   