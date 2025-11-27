import json
from typing import Any, Dict, Optional, Iterator, Tuple, Union
from sentence_transformers import SentenceTransformer
import redis
from config.config import config
from utils.utils import LogUtils

logger = LogUtils().get_logging()


class StorageModel:
    """
    StorageModel provides an abstraction for storing and retrieving question-answer pairs
    with sentence embeddings, supporting both in-memory (dict) and Redis storage.

    Attributes:
        name (str): Name for the storage instance.
        type (str): Type of backend used ("memory" or "redis").
        encoder (SentenceTransformer): Embedding model.
        cache (Union[Dict[str, Dict[str, Any]], redis.Redis]): Underlying storage backend.
        ttl (int): Time-to-live for entries, in seconds (only used with Redis).
    """

    def __init__(
        self,
        name: str = "semantic-cache",
        type: str = "memory",
        encoder_model: str = "all-mpnet-base-v2",
        ttl: int = 3600,
    ) -> None:
        """
        Initialize StorageModel.

        Args:
            name (str): Name of storage for namespacing.
            type (str): Storage backend ("memory" or "redis").
            encoder_model (str): SentenceTransformer model name.
            ttl (int): Time-to-live for Redis cache entries, in seconds.
        """
        self.name: str = name
        self.type: str = type
        self.encoder: SentenceTransformer = SentenceTransformer(encoder_model)
        self.cache: Union[Dict[str, Dict[str, Any]], redis.Redis, None] = None
        if self.type == "redis":
            self.cache = redis.Redis(
                host=config["redis_host"], port=int(config["redis_port"]), db=0
            )
        else:
            self.cache = {}
        self.ttl: int = ttl
        logger.info("{} Storage set".format(type))
        logger.info("{} Encoder_model set".format(encoder_model))

    def store(self, input_text: str, output_data: Any) -> None:
        """
        Store an input text, its embedding, and output data in the cache.

        Args:
            input_text (str): The question key.
            output_data (Any): The answer or value to store.
        """
        embedding = self.encoder.encode([input_text])[0]
        data: json = json.dumps(
            {
                "question": input_text,
                "embedding": embedding.tolist(),
                "answer": output_data,
            }
        )
        if self.type == "redis":
            self.cache.set(input_text, data, ex=self.ttl)
        else:
            self.cache[input_text] = data

    def get_all_key_values(self) -> Iterator[Tuple[str, Dict[str, Any]]]:
        """
        Get all key-value pairs in the cache.

        Returns:
            Iterator over (question, data dictionary) pairs.
        """
        if self.type == "redis":
            result_dict: Dict[str | json] = {}
            for key in self.cache.scan_iter():
                if key is not None:
                    key_str = key.decode("utf-8")
                    value_bytes = self.cache.get(key)
                    if value_bytes is not None:
                        try:
                            value_str = value_bytes.decode("utf-8")
                            result_dict[key_str] = value_str
                        except Exception as e:
                            logger.error(
                                f"Error deserializing redis value for {key}: {e}"
                            )

            return result_dict

        else:
            return self.cache.items()

    def get_value_by_key(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the cached data for a given key.

        Args:
            key (str): The question string key.

        Returns:
            Optional[Dict[str, Any]]: The stored dictionary or None.
        """
        if self.type == "redis":
            value_bytes = self.cache.get(key)
            if value_bytes is not None:
                try:
                    return eval(value_bytes.decode("utf-8"))
                except Exception as e:
                    logger.error(f"Error deserializing redis value for {key}: {e}")
                    return None
            return None
        else:
            value = self.cache.get(key)
            if value is not None:
                try:
                    return json.loads(value)
                except Exception as e:
                    logger.error(f"Error deserializing redis value for {key}: {e}")
                    return None
            return None
