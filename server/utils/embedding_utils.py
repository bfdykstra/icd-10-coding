import asyncio
from functools import partial
from sentence_transformers import SentenceTransformer
from utils.config import config
import numpy as np

embedding_model = SentenceTransformer(config.embedding_model)


async def encode_async(text: str) -> np.ndarray:
  loop = asyncio.get_event_loop()
  return await loop.run_in_executor(None, embedding_model.encode, text)