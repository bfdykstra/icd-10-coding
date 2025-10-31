import os
import uvicorn

# Set tokenizers parallelism to avoid warnings when using uvicorn with reload
# This must be set before any tokenizers are imported/initialized
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload for development
        log_level="info"
    ) 