import getpass
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def load_openai_key():
    if not os.getenv("OPENAI_API_KEY"):
        api_key = getpass.getpass("Enter your OpenAI API key: ")
        os.environ["OPENAI_API_KEY"] = api_key
    else:
        print("> OpenAI API key is already loaded in the environment")


config = dict(
    name="default-cache",
    storage_type="memory",  # storage options: memory, redis
    reranking=False,
    llm_type="local",  # llm options: local or openAI
    redis_host=os.getenv("REDIS_HOST", "localhost"),
    redis_port=os.getenv("REDIS_PORT", "6379"),
    cache_name=os.getenv("CACHE_NAME", "semantic-cache"),
    similarity_threshold=float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.3")),
    ttl_seconds=os.getenv("CACHE_TTL_SECONDS", "3600"),
    encode_model="all-mpnet-base-v2",
    ollama_host=os.getenv("OLLAMA_HOST", "localhost"),
    ollama_port=os.getenv("OLLAMA_PORT", "11434"),
)
